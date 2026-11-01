"""Past-only retrieval of ordered story neighborhoods; no future payoff labels.

Unlike bag-of-events similarity, [old, recent, forming] are different slots.
Candidates must match scale, phase and context mask, and end before the entire
query neighborhood begins. No learned metric or claim of predictive value.
"""
import numpy as np
import pandas as pd

from .tokens import NS, TICK, quote_arrays

DESCRIPTOR_NAMES = ['net_in_amplitudes', 'range_in_amplitudes', 'log_path_in_amplitudes',
    'signed_efficiency', 'log_elapsed_in_limits', 'retreat_in_amplitudes',
    'rebound_in_amplitudes', 'quarter_1', 'quarter_2', 'quarter_3', 'direction']


def descriptors(motion, quotes):
    """Raw motion updates already contain exact running extrema/path sums."""
    p = motion.mid_nano.to_numpy(np.int64)
    anchor = motion.start_mid_nano.to_numpy(np.int64)
    scale = motion.amplitude_ticks.to_numpy(float)*TICK
    path = motion.path_nano.to_numpy(float)
    dt = motion.available_ns.to_numpy(np.int64)-motion.start_ns.to_numpy(np.int64)
    qt, price, _, _ = quote_arrays(quotes)
    shape = []
    for k in (1, 2, 3):
        clock = motion.start_ns.to_numpy(np.int64)+dt*k//4
        idx = np.minimum(motion.source_index.to_numpy(),
            np.maximum(motion.start_index.to_numpy(), np.searchsorted(qt, clock, side='right')-1))
        shape.append((price[idx]-anchor)/scale)
    return np.column_stack([(p-anchor)/scale,
        (motion.high_nano.to_numpy()-motion.low_nano.to_numpy())/scale,
        np.log1p(path/scale), np.divide(p-anchor, path, out=np.zeros(len(p)), where=path>0),
        np.log1p(dt/NS/motion.time_limit_seconds.to_numpy()),
        (motion.high_nano.to_numpy()-p)/scale, (p-motion.low_nano.to_numpy())/scale,
        *shape, motion.direction.to_numpy()]).astype(np.float32)


def ordered_distance(query, candidates, mask):
    q = np.asarray(query, float)
    c = np.asarray(candidates, float)
    m = np.broadcast_to(np.asarray(mask, bool)[..., None], q.shape)
    if not m.any(): raise ValueError('At least one observed story required')
    return np.mean((c[:, m]-q[m])**2, axis=1)


class StoryIndex:
    def __init__(self, sessions):
        frames, vectors = [], []
        for session in sessions:
            frames.append(session.motion)
            vectors.append(descriptors(session.motion, session.quotes))
        self.frame = pd.concat(frames, ignore_index=True)
        self.vectors = np.concatenate(vectors)
        self.lookup = {token: i for i, token in enumerate(self.frame.token_id)}
        self.completed = {}
        for key, group in self.frame.groupby(['session_id','scale','segment'], sort=False):
            ids = group.index[group.phase.isin(['CONFIRMED','TIME_LIMIT']) & group.elapsed_seconds.gt(0)].to_numpy()
            self.completed[key] = ids[np.argsort(self.frame.loc[ids, 'available_ns'].to_numpy(), kind='stable')]
        self.context = np.zeros((len(self.frame), 3, len(DESCRIPTOR_NAMES)), np.float32)
        self.mask = np.zeros((len(self.frame), 3), bool)
        self.context_start = self.frame.start_ns.to_numpy(np.int64).copy()
        self.parents = np.full((len(self.frame), 2), -1, dtype=np.int64)
        self.context[:, 2] = self.vectors
        self.mask[:, 2] = True
        for key, group in self.frame.groupby(['session_id','scale','segment'], sort=False):
            ids = group.index.to_numpy()
            history = self.completed[key]
            if not len(history): continue
            ends = self.frame.loc[history, 'available_ns'].to_numpy(np.int64)
            positions = np.searchsorted(ends, group.start_ns.to_numpy(np.int64), side='right')
            for slot in (0, 1):
                pos = positions-(2-slot)
                ok = pos >= 0
                selected = history[np.maximum(pos, 0)]
                self.context[ids[ok], slot] = self.vectors[selected[ok]]
                self.mask[ids[ok], slot] = True
                self.parents[ids[ok], slot] = selected[ok]
                self.context_start[ids[ok]] = np.minimum(self.context_start[ids[ok]],
                    self.frame.loc[selected[ok], 'start_ns'].to_numpy(np.int64))

    def query(self, token, session, previous_session_id, *, per_session=3):
        q = pd.DataFrame([token])
        vector = descriptors(q, session.quotes)[0]
        context = np.zeros((3, len(DESCRIPTOR_NAMES)), np.float32)
        mask = np.array([False, False, True])
        context[2] = vector
        start = int(token.start_ns)
        parents = []
        history = self.completed.get((token.session_id, token.scale, token.segment), np.array([], int))
        history = history[self.frame.loc[history, 'available_ns'].to_numpy() <= start][-2:]
        for slot, idx in zip(range(2-len(history), 2), history):
            context[slot] = self.vectors[idx]; mask[slot] = True
            start = min(start, int(self.frame.loc[idx, 'start_ns']))
            parents.append(self.frame.loc[idx, 'token_id'])
        f = self.frame
        eligible = (f.scale.eq(token.scale) & f.phase.eq(token.phase)
                    & f.available_ns.lt(start) & f.elapsed_seconds.gt(0)
                    & f.session_id.isin([token.session_id, previous_session_id])
                    & (self.mask == mask).all(axis=1))
        result = []
        for date in (token.session_id, previous_session_id):
            if date is None: continue
            ids = f.index[eligible & f.session_id.eq(date)].to_numpy()
            if not len(ids): continue
            distances = ordered_distance(context, self.context[ids], mask)
            order = np.argsort(distances, kind='stable')
            selected_intervals = []
            seen_stories = set()
            for ix in order:
                i = ids[ix]; row = f.iloc[i]
                interval = (int(self.context_start[i]), int(row.available_ns))
                if row.story_id in seen_stories or any(interval[0] <= b and a <= interval[1] for a,b in selected_intervals):
                    continue
                seen_stories.add(row.story_id); selected_intervals.append(interval)
                result.append(dict(query_token_id=token.token_id, token_id=row.token_id,
                    session_id=date, scale=row.scale, phase=row.phase, distance=float(distances[ix]),
                    context_start_ns=interval[0], available_ns=interval[1],
                    query_context_start_ns=start, query_available_ns=int(token.available_ns),
                    context_token_ids=[f.iloc[p].token_id for p in self.parents[i] if p>=0]+[row.token_id],
                    query_context_token_ids=parents+[token.token_id],
                    interpretation='Measured geometric resemblance; not a probability or a future-outcome match'))
                if len(selected_intervals) >= per_session: break
        return result
