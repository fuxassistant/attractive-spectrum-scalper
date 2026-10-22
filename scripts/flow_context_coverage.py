"""Pre-fit coverage audit; no target-dependent selection or model fitting."""
from pathlib import Path
import json
import hashlib
import numpy as np
import pandas as pd
from spectrum.cognition.episodes import NS
from spectrum.cognition.execution import QuoteTape
from spectrum.playground.data import MonthDay, QUOTE_CACHE

REPORT = Path('reports/flow_context_v2')
SOURCE = Path('data/derived/scalper_v1')
MARKET = Path('data/derived/basket_multimonth_v1')


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(4 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def save(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8', newline='\n')


def main():
    manifest = json.loads((SOURCE / 'manifest.json').read_text())
    probes = {'2025-10-01', '2025-11-03', '2025-12-01', '2026-01-02', '2026-01-30',
              '2026-02-02', '2026-02-06', '2026-02-16', '2026-02-23', '2026-02-27'}
    days, checks, hashes = [], [], {}
    for date in manifest['days']:
        if date[:7] not in ('2025-10', '2025-11', '2025-12', '2026-01', '2026-02'):
            continue
        path = SOURCE / (date + '.parquet'); receipt = json.loads(path.with_suffix('.json').read_text())
        actual = sha(path)
        if actual != receipt['sha256']:
            raise ValueError('Source changed: ' + date)
        hashes[date] = actual
        d = pd.read_parquet(path, columns=['t_ns', 'sigma_known', 'sigma600', 'spread_ticks'])
        d = d.loc[(d.t_ns // NS) % 30 == 0]
        if d.empty:
            days.append(dict(date=date, excluded='NO_SOURCE_DECISIONS', clock_roots=0)); continue
        meta = json.loads((MARKET / date[:7] / date / 'manifest.json').read_text())
        sigma = (d.sigma_known > 0) & (d.sigma600 >= .3)
        positive = sigma & (d.spread_ticks > 0)
        clock = d.t_ns + 335 * NS < int(meta['liquidation_deadline_ns'])
        uncapped = positive & clock; legacy = uncapped & (d.spread_ticks <= 2)
        bins = np.select([d.spread_ticks <= 2, d.spread_ticks <= 4, d.spread_ticks <= 8], ['0-2', '2-4', '4-8'], '8+')
        days.append(dict(date=date, symbol=meta['symbol'], clock_roots=len(d), sigma_pass=int(sigma.sum()),
            positive_spread=int(positive.sum()), legacy_before_deadline=int((positive & (d.spread_ticks <= 2)).sum()),
            uncapped_roots=int(uncapped.sum()), legacy_roots=int(legacy.sum()),
            spread_quantiles={str(p): float(d.spread_ticks.quantile(p)) for p in (.1, .5, .9, .99)},
            spread_bins={name: int(((bins == name) & uncapped).sum()) for name in ('0-2', '2-4', '4-8', '8+')}))
        if date in probes:
            day = MonthDay(MARKET / date[:7] / date, verify=True); q = QuoteTape.from_day(day)
            t = d.t_ns.to_numpy(np.int64); j = np.searchsorted(q.quote_t, t, side='right') - 1
            if (j < 0).any() or not q.valid[j].all():
                raise ValueError('Unsupported cached source root: ' + date)
            spreads = (q.ask[j] - q.bid[j]) / 1e8
            errors = np.abs(spreads - d.spread_ticks.to_numpy())
            if np.any(errors > 1e-6):
                raise ValueError('Source spread differs from BBO: ' + date)
            qhash = {}
            for part in day.meta['quote_parts']:
                file = Path(part['root']) / 'quotes.parquet'; h = sha(file)
                if h != part['quote_file']['sha256']:
                    raise ValueError('Quote hash changed')
                qhash[str(file)] = h
            checks.append(dict(date=date, roots=len(t), spread_max_abs_error=float(errors.max()),
                               all_asof_quotes_valid=True, source_quote_hashes=qhash))
            QUOTE_CACHE.clear()
    monthly = {}
    for month in ('2025-10', '2025-11', '2025-12', '2026-01', '2026-02'):
        rows = [r for r in days if r['date'][:7] == month and not r.get('excluded')]
        sums = {k: sum(r[k] for r in rows) for k in ('clock_roots', 'sigma_pass', 'positive_spread', 'legacy_before_deadline', 'uncapped_roots', 'legacy_roots')}
        monthly[month] = dict(days=len(rows), **sums, median_of_daily_median_spread=float(np.median([r['spread_quantiles']['0.5'] for r in rows])),
            spread_bins={b: sum(r['spread_bins'][b] for r in rows) for b in ('0-2', '2-4', '4-8', '8+')})
    result = dict(status='PASS', phase='BEFORE_V2_FIT', days=days, monthly=monthly, quote_checks=checks,
                  source_sha256=hashes, script_sha256=sha(__file__),
                  conclusion='February source rows exist. The <=2 tick spread gate removes almost all roots; positive-spread uncapped cohort is a new experiment, not a correction of v1.')
    save(REPORT / 'coverage_audit.json', result)
    print(json.dumps(dict(monthly=monthly, checked_quote_roots=sum(r['roots'] for r in checks), source_files=len(hashes)), indent=2))


if __name__ == '__main__':
    main()
