import json

import numpy as np
import pandas as pd

from spectrum.player_v3.report import SERIES_FIELDS, _read_replay, _risk_contract, _series_column, build_report


def test_series_contract_and_floor_are_read_from_saved_state():
    replay = {'decisions': {
        't_ns': [1, 2, 3, 4], 'series_floor': [0., 100., 100., 120.],
        'series_start_equity': [0., 200., 200., 240.], 'series_id': [0, 1, 1, 2],
        'series_loss_fraction': [0., .5, .501, .05],
        'series_active': [False, True, False, True], 'risk_dead': [False, False, True, False],
    }}
    protocol = {'risk_contract': {'version': 'SERIES_LOSS_GT_HALF_V1', 'initial_capital': 200.,
                                  'loss_limit_fraction': .5, 'comparison': '>'}}
    risk = _risk_contract(protocol, [replay])
    assert risk['mode'] == 'SERIES'
    assert risk['comparison'] == '>'
    assert risk['loss_limit_fraction'] == .5
    np.testing.assert_allclose(_series_column(replay['decisions'], 'series_floor'),
                               [np.nan, 100., 100., 120.], equal_nan=True)
    # The display preserves exact-half alive and strict-breach dead flags.
    assert not replay['decisions']['risk_dead'][1]
    assert replay['decisions']['risk_dead'][2]


def test_no_series_state_is_explicitly_legacy_not_reconstructed():
    risk = _risk_contract({'version': 'MIXED_PLAYER_V3', 'contract': {'initial_capital': 200.}},
                          [{'decisions': {'t_ns': [1], 'equity_before': [100.]}}])
    assert risk['mode'] == 'LEGACY'
    assert risk['loss_limit_fraction'] is None
    assert 'no series floor is reconstructed' in risk['legacy']


def test_actual_snapshot_schema_keeps_censor_and_before_state(tmp_path):
    for prefix in ('', 'risk_'):
        path = tmp_path/'replays'/('prefixed' if prefix else 'neural')/'2026-05-18'
        path.mkdir(parents=True)
        raw = pd.DataFrame(dict(t_ns=np.asarray([1779094800000000000, 1779094860000000000, 1779094920000000000]),
                                equity=[149., 90., 99.]))
        values = dict(series_id=[1, 1, 1], series_start_equity=[200., 200., 200.],
                      series_floor=[100., 100., 100.], series_active=[True, False, False],
                      series_loss_fraction=[.25, .55, .505], series_status=['active', 'censored', 'ruined'],
                      series_cumulative_net=[-50., -110., -101.], liquidation_equity=[150., 90., 99.],
                      series_start_ns=['1779094800123456789']*3)
        for key, value in values.items():
            raw[prefix+key] = value
        raw[prefix+'series_start_equity_after'] = [300., 300., 300.]
        raw[prefix+'series_status_after'] = ['ruined']*3
        raw.to_parquet(path/'decisions.parquet', index=False)
        replay = _read_replay(tmp_path, path/'decisions.parquet')
        d = replay['decisions']
        assert d['series_start_equity'] == [200., 200., 200.]
        assert d['equity_before'] == [150., 90., 99.]
        assert d['equity_after'] == [149., 90., 99.]
        assert d['series_status'] == ['active', 'censored', 'ruined']
        assert d['risk_dead'] == [False, False, True]
        assert d['series_net'] == [-50., -110., -101.]
        assert d['series_start_ns'] == ['1779094800123456789']*3
        assert not any(key.endswith('_after') for key in d if key != 'equity_after')
        pd.testing.assert_frame_equal(pd.read_parquet(path/'decisions.parquet'), raw)


def test_generated_report_preserves_dynamic_floor_and_exact_clocks(tmp_path):
    path = tmp_path/'replays'/'mixed_full_s61018_sample101'/'2026-05-18'
    path.mkdir(parents=True)
    t = 1779094800000000000 + np.arange(6, dtype=np.int64)*60_000_000_000
    pd.DataFrame(dict(t_ns=t, mid=[4550., 4551., 4552., 4551., 4552., 4553.],
        action=[0, 1, 0, 0, 1, 0], equity_before=[200., 200., 100., 199., 240., 125.],
        equity_after=[200., 100., 199., 240., 125., 119.],
        series_floor=[0., 100., 100., 100., 120., 120.],
        series_start_equity=[0., 200., 200., 200., 240., 240.], series_id=[0, 1, 1, 1, 2, 2],
        series_loss_fraction=[0., 0., .5, .005, 0., 115/240],
        series_active=[False, True, True, True, True, True], risk_dead=[False]*6,
        policy_evaluated=[True]*6)).to_parquet(path/'decisions.parquet', index=False)
    (tmp_path/'contract.json').write_text(json.dumps({'risk_contract': {
        'version': 'SERIES_LOSS_GT_HALF_V1', 'initial_capital': 200.,
        'loss_limit_fraction': .5, 'comparison': '>'}}), encoding='utf-8')
    receipt = build_report(tmp_path)
    html = (tmp_path/'index.html').read_text(encoding='utf-8')
    raw = html.split('<script id="audit-data" type="application/json">')[1].split('</script>')[0]
    data = json.loads(raw)
    assert data['risk_contract']['mode'] == 'SERIES'
    assert all(name in data['replays'][0]['decisions'] for name in SERIES_FIELDS)
    assert data['replays'][0]['decisions']['t_ns'] == [str(int(v)) for v in t]
    assert data['replays'][0]['decisions']['series_floor'][-1] == 120.
    assert 'ликвидационный барьер — $100' not in html
    assert 'vals.push(100,200)' not in html
    assert receipt['risk_contract']['comparison'] == '>'
    assert receipt['figures']
