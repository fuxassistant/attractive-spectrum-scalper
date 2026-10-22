"""Verify the preserved namespace migration changed IDs only, not factual values."""
import json
from pathlib import Path
import pandas as pd
from scripts.crossing_map_build import OUT, REPORT, DATES
from scripts.flow_context_coverage import save, sha


def main():
    previous = OUT.with_name(OUT.name + '_superseded_ids')
    old_manifest = json.loads((previous / 'manifest.json').read_text())
    new_manifest = json.loads((OUT / 'manifest.json').read_text())
    assert old_manifest['status'] == new_manifest['status'] == 'COMPLETE'
    tables = []
    links = []
    for date in DATES:
        for kind in ['bank', 'map', 'roots', 'scene']:
            name = date + '_' + kind + '.parquet'
            before = pd.read_parquet(previous / name)
            after = pd.read_parquet(OUT / name)
            old_day = next(d for d in old_manifest['days'] if d['date'] == date)
            new_day = next(d for d in new_manifest['days'] if d['date'] == date)
            assert sha(previous / name) == old_day['files'][name]
            assert sha(OUT / name) == new_day['files'][name]
            excluded = ['event_id'] if kind in ['bank', 'map'] else []
            pd.testing.assert_frame_equal(before.drop(columns=excluded),
                                          after.drop(columns=excluded), check_exact=True)
            tables.append(dict(file=name, rows=len(after), excluded_columns=excluded,
                               exact_numeric_and_non_id_parity=True,
                               before_sha256=sha(previous / name), after_sha256=sha(OUT / name)))
        bank = pd.read_parquet(OUT / (date + '_bank.parquet')).set_index('event_id')
        mapping = pd.read_parquet(OUT / (date + '_map.parquet'))
        assert bank.index.is_unique
        observed = mapping.loc[mapping.cross_ordinal >= 0]
        matched = bank.loc[observed.event_id]
        for left, right in [('family', 'family'), ('scale', 'scale'), ('role', 'role'),
                            ('cross_ordinal', 'ordinal'), ('cross_ns', 't_ns'),
                            ('cross_direction', 'direction')]:
            assert (observed[left].to_numpy() == matched[right].to_numpy()).all()
        assert (mapping.loc[mapping.cross_ordinal < 0, 'event_id'] == '').all()
        links.append(dict(date=date, resolved_cells=len(observed), bank_events=len(bank),
                          identity_link_checks='PASS'))
    result = dict(status='PASS', preserved_previous=str(previous),
                  reason='Physical event identity must not depend on target/loss/query code',
                  compared_tables=len(tables), tables=tables, bank_map_links=links,
                  previous_manifest_sha256=sha(previous / 'manifest.json'),
                  current_manifest_sha256=sha(OUT / 'manifest.json'), script_sha256=sha(__file__))
    save(REPORT / 'identity_migration_receipt.json', result)
    print(json.dumps(dict(status='PASS', exact_table_parity=len(tables),
                          resolved_id_links=sum(x['resolved_cells'] for x in links))), flush=True)


if __name__ == '__main__':
    main()
