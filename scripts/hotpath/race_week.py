"""Race map for one test week: long/short scores, opportunity, smoothness against the sign-flipped null.

Decisions: every second of the London open .. NY close session with a known sigma600. For each magnitude G
of race.GRID_G the real race and three sign-flipped null races are computed. Smoothness is measured on the
winner label series: run lengths of the same winner, flips per hour, autocorrelation of the dominance score,
and episodes (runs of one winner lasting at least 30 s) per hour. Writes a JSON summary and the page data.
"""
import json
import sys
from pathlib import Path
import numpy as np
from spectrum.cognition import race as rc
from spectrum.cognition.hotpath import base_day

DAYS = ('2025-11-17', '2025-11-18', '2025-11-19', '2025-11-20', '2025-11-21')
OUT = Path('reports/race_v1')
STEP = 5
SEEDS = (11, 12, 13)
EPISODE_S = 30


def acf(x, lag):
    x = np.asarray(x, float); m = np.isfinite(x[lag:]) & np.isfinite(x[:-lag])
    a, b = x[lag:][m], x[:-lag][m]
    return float(np.corrcoef(a, b)[0, 1]) if len(a) > 10 and a.std() > 0 and b.std() > 0 else np.nan


def metrics(r, hours):
    w = r['winner']; valid = w != rc.CENSORED
    wv = np.where(valid, w, 0)
    rl = rc.runs(wv); ep = rl[rl >= EPISODE_S]
    flips = int(np.sum((wv[1:] != wv[:-1]) & valid[1:] & valid[:-1]))
    won = np.isin(w, (1, -1))
    return dict(long=float((w == 1)[valid].mean()), short=float((w == -1)[valid].mean()), none=float((w == 0)[valid].mean()),
                censored=float((~valid).mean()), clean=float(r['clean'][won].mean()) if won.any() else np.nan,
                run_mean=float(rl.mean()) if len(rl) else 0., run_median=float(np.median(rl)) if len(rl) else 0.,
                flips_per_hour=flips / hours, episodes_per_hour=len(ep) / hours, episode_mean_s=float(ep.mean()) if len(ep) else 0.,
                acf5=acf(r['dominance'], 5), acf30=acf(r['dominance'], 30), acf120=acf(r['dominance'], 120),
                H_median=float(np.median(r['H'])), t_win_median=float(np.median(r['t_win'][won])) if won.any() else np.nan,
                opp_mean=float(np.nanmean(r['opp'])), opp_share=float(np.nanmean(r['opp'] > 0)))


def main(days=DAYS):
    OUT.mkdir(parents=True, exist_ok=True); summary = []; page = dict(grid=list(rc.GRID_G), step=STEP, costs=dict(win=rc.C_WIN, loss=rc.C_LOSS),
                                                                      clean=rc.CLEAN, days=[])
    for date in days:
        b, m = base_day(date)
        k = np.arange(m['london'], m['ny_close'])
        k = k[b['sigma_known'][k] & np.isfinite(b['mid'][k]) & (b['sigma'][k] > 0)]
        sig = b['sigma'][k]; hours = len(k) / 3600
        nulls = [rc.sign_flipped(b['mid'], b['seg'], np.random.default_rng(s * 100000 + int(date.replace('-', '')) % 100000)) for s in SEEDS]
        sub = np.arange(0, len(k), STEP)
        day = dict(date=date, t=((k[sub] - m['london'])).tolist(), mid=np.round(b['mid'][k[sub]] / 10, 2).tolist(),
                   sigma=np.round(sig[sub], 3).tolist(), G={})
        for G in rc.GRID_G:
            r = rc.race(b['mid'], b['seg'], k, sig, G)
            summary.append(dict(date=date, G=G, variant='real', **metrics(r, hours)))
            for s, nm in zip(SEEDS, nulls):
                rn = rc.race(nm, b['seg'], k, sig, G)
                summary.append(dict(date=date, G=G, variant=f'null{s}', **metrics(rn, hours)))
            code = np.where(r['winner'] == rc.CENSORED, 0, r['winner'] * np.where(r['clean'], 2, 1))
            rn = rc.race(nulls[0], b['seg'], k, sig, G)
            ncode = np.where(rn['winner'] == rc.CENSORED, 0, rn['winner'] * np.where(rn['clean'], 2, 1))
            day['G'][str(G)] = dict(code=code[sub].astype(int).tolist(), opp=np.round(np.nan_to_num(r['opp'][sub]), 2).tolist(),
                                    long=np.round(np.nan_to_num(r['long'][sub]), 2).tolist(), short=np.round(np.nan_to_num(r['short'][sub]), 2).tolist(),
                                    H=r['H'][sub].astype(int).tolist(), null_code=ncode[sub].astype(int).tolist(),
                                    null_mid=np.round(nulls[0][k[sub]] / 10, 2).tolist() if G == rc.GRID_G[0] else None)
        page['days'].append(day)
        print(date, 'done', flush=True)
    (OUT / 'race_week_summary.json').write_text(json.dumps(summary, indent=1), encoding='utf8')
    (OUT / 'race_week_page.json').write_text(json.dumps(page, separators=(',', ':')), encoding='utf8')


if __name__ == '__main__':
    main(tuple(sys.argv[1:]) or DAYS)
