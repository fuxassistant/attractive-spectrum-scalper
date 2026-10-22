"""Compact the race-week output into the page data file (reports/race_v1/race_data.json)."""
import json
from pathlib import Path
import numpy as np
import pandas as pd

SRC = Path('reports/race_v1')
CODE = {-2: 'a', -1: 'b', 0: 'c', 1: 'd', 2: 'e'}


def main():
    page = json.loads((SRC / 'race_week_page.json').read_text(encoding='utf8'))
    s = pd.DataFrame(json.loads((SRC / 'race_week_summary.json').read_text(encoding='utf8')))
    s['kind'] = np.where(s.variant == 'real', 'real', 'null')
    cols = ['long', 'short', 'none', 'clean', 'H_median', 't_win_median', 'run_mean', 'run_median', 'flips_per_hour',
            'episodes_per_hour', 'episode_mean_s', 'acf5', 'acf30', 'acf120', 'opp_share', 'opp_mean']
    agg = s.groupby(['G', 'kind'])[cols].mean().round(4)
    summary = []
    for G in page['grid']:
        summary.append(dict(G=G, real=agg.loc[(G, 'real')].to_dict(), null=agg.loc[(G, 'null')].to_dict()))
    days = []
    for d in page['days']:
        out = dict(date=d['date'], t=d['t'], mid=d['mid'], sig=d['sigma'], G={})
        out['nmid'] = d['G'][str(page['grid'][0])]['null_mid']
        for G in page['grid']:
            g = d['G'][str(G)]
            out['G'][str(G)] = dict(c=''.join(CODE[x] for x in g['code']), nc=''.join(CODE[x] for x in g['null_code']),
                                    L=[int(round(v * 100)) for v in g['long']], S=[int(round(v * 100)) for v in g['short']],
                                    O=[int(round(v * 100)) for v in g['opp']])
        days.append(out)
    data = dict(grid=page['grid'], costs=page['costs'], clean=page['clean'], k=2, h_min=20, h_max=900, days=days, summary=summary)
    (SRC / 'race_data.json').write_text(json.dumps(data, separators=(',', ':')), encoding='utf8')
    print((SRC / 'race_data.json').stat().st_size)


if __name__ == '__main__':
    main()
