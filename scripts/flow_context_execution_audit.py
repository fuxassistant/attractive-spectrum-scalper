"""Apply the independent V1 MBO/exit audit to V2's broader frozen population."""
import json
import pandas as pd
from scripts import opportunity_episodes_audit as audit
from scripts.flow_context_coverage import save, sha
from spectrum.cognition.flow_context_lab import DATA, REPORT
from spectrum.cognition import episodes as ep


def candidates(days):
    scenes = []; seen = set(); cats = set(); buckets = set()
    for date in days:
        if date[:7] not in ('2026-01', '2026-02'):
            continue
        f = pd.read_parquet(DATA / (date + '.parquet'))
        f = f.loc[f['mode'] == 'P'].sort_values(['t_ns', 'side'], ascending=[True, False])
        selections = []
        if date[:7] == '2026-02':
            spread = f.spread_ticks
            for name, mask in {'0-2': spread <= 2, '2-4': (spread > 2) & (spread <= 4), '4-8': (spread > 4) & (spread <= 8), '8+': spread > 8}.items():
                if name not in buckets and mask.any():
                    selections.append((f.loc[mask].iloc[0], 'first_Feb_spread_' + name)); buckets.add(name)
        for cat in (0, 1, 2, 3, -1):
            if cat not in cats and (f.category == cat).any():
                selections.append((f.loc[f.category == cat].iloc[0], 'first_' + (ep.CLASSES[cat] if cat >= 0 else 'CENSORED'))); cats.add(cat)
        for row, reason in selections:
            key = (date, int(row.t_ns), int(row.side))
            if key in seen:
                continue
            seen.add(key); scenes.append(dict(date=date, t_ns=str(key[1]), side=key[2], mode='P', selection=reason))
    return scenes, dict(missing_categories=[ep.CLASSES[c] if c >= 0 else 'CENSORED' for c in (0, 1, 2, 3, -1) if c not in cats])


if __name__ == '__main__':
    # Process-local dependency injection: the original V1 files are untouched.
    audit.DATA, audit.REPORT = DATA, REPORT
    audit.save = save; audit.scene_candidates = candidates
    audit.main()
    path = REPORT / 'execution_audit.json'; result = json.loads(path.read_text())
    result['reused_audit_code_sha256'] = sha(audit.__file__)
    result['v2_driver_sha256'] = sha(__file__)
    save(path, result)
