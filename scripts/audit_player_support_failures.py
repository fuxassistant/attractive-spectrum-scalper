"""Read-only attribution of saved player support stops, including raw evidence."""
from collections import Counter
from pathlib import Path
import json
import numpy as np
from spectrum.ingest import save_json
from spectrum.source import DbnSource


def main():
    root = Path('reports/player_v3/series_001')
    out = Path('reports/player_v3/support_investigation_001')
    out.mkdir(exist_ok=True)
    stops = []
    for path in sorted((root / 'models').glob('*/epoch_*.json')):
        obj = json.loads(path.read_text())
        for week, result in enumerate(obj['train']):
            if result['censored']:
                stops.append(dict(model=path.parent.name, epoch=path.stem, week=week,
                                  date=result['days'][-1]['date'], ns=int(result['censor_ns'])))
    summary = json.loads((root / 'summary.json').read_text())
    for model in summary['models']:
        for result in model['stochastic_evaluation']:
            if result['censored']:
                stops.append(dict(model=model['variant'], simulation=result['simulation_seed'],
                                  date=result['days'][-1]['date'], ns=int(result['censor_ns'])))
    evidence = {}
    for date in sorted({s['date'] for s in stops}):
        folder = Path('data/research/player_v3/memory_001') / date
        a = {name: np.load(folder / (name+'.npy'), mmap_mode='r')
             for name in ('quote_t', 'source_t', 'valid', 'epoch', 'source_ordinal')}
        for ns in sorted({s['ns'] for s in stops if s['date'] == date}):
            j = int(np.searchsorted(a['quote_t'], ns))
            assert int(a['quote_t'][j]) == ns and j > 0
            gap = int(a['quote_t'][j] - a['quote_t'][j-1])
            row = dict(date=date, ns=str(ns), index=j, gap_ns=gap,
                       same_epoch=bool(a['epoch'][j] == a['epoch'][j-1]),
                       endpoints_valid=bool(a['valid'][j] and a['valid'][j-1]),
                       source_age_ns=int(a['quote_t'][j]-a['source_t'][j]),
                       source_ordinals=[int(a['source_ordinal'][j-1]), int(a['source_ordinal'][j])])
            row['classification'] = ('PLAYER_ONLY_5S_GAP_WITHIN_CORE_30S'
                if 5_000_000_000 < gap <= 30_000_000_000 and row['same_epoch']
                and row['endpoints_valid'] and row['source_age_ns'] == 0 else 'REQUIRES_OTHER_SUPPORT_EVIDENCE')
            evidence[(date, ns)] = row
    for stop in stops:
        stop['classification'] = evidence[(stop['date'], stop['ns'])]['classification']
        stop['ns'] = str(stop['ns'])
    result = dict(status='DIAGNOSIS_ONLY_NOT_TRAINING_ADMISSION',
                  population_stops=len(stops), counts=dict(Counter(s['classification'] for s in stops)),
                  unique_stop_points=list(evidence.values()), stops=stops,
                  limitation='Valid endpoints do not alone prove source continuity; raw interval checked separately.')
    save_json(out / 'saved_stop_attribution.json', result)
    print(json.dumps(result['counts']), flush=True)
    config = json.loads(Path('data/derived/full_mbo_v1/run_config.json').read_text())
    member = next(m for m in config['members'] if m['key'] == 'member_007')
    first, last = 51800827, 51800831
    raw_rows = []; offset = 0
    for block in DbnSource(member['archive'], member['member']):
        lo, hi = max(0, first-offset), min(len(block), last-offset+1)
        for i in range(lo, hi):
            row = {name: (block[name][i].decode() if block.dtype[name].kind == 'S'
                          else str(int(block[name][i]))) for name in block.dtype.names}
            raw_rows.append(dict(source_ordinal=str(offset+i), **row))
        offset += len(block)
        if offset > last:
            break
    assert len(raw_rows) == last-first+1
    save_json(out / 'raw_interval.json', dict(
        archive=member['archive'], member=member['member'], rows=raw_rows,
        bounds='Exact original DBN source ordinals; no reconstruction or training was changed',
        full_source_hash_verified_this_scan=False))
    print(json.dumps(dict(raw_rows=len(raw_rows), status='RAW_INTERVAL_EXTRACTED')), flush=True)


if __name__ == '__main__':
    main()
