# -*- coding: utf-8 -*-
"""Расширенный подбор гиперпараметров по walk-forward RPS."""
import sys, os, itertools
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from backtest import walk_forward, summarize

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

if __name__ == '__main__':
    df = pd.read_csv(os.path.join(ROOT, 'data', 'matches.csv'))
    grid = dict(
        half_life=[110, 150, 200, 260],
        alpha=[0.15, 0.3, 0.5, 0.7, 1.0],
        w_goals=[0.0, 0.15, 0.30],
        promoted_shift=[0.0, 0.25, 0.40],
    )
    keys = list(grid)
    rows = []
    for combo in itertools.product(*[grid[k] for k in keys]):
        params = dict(zip(keys, combo))
        r = walk_forward(df, params)
        s = summarize(r)
        rows.append({**params, 'n': s['n'], 'logloss': s['logloss'], 'rps': s['rps'],
                     'rps_book': s.get('rps_book'), 'logloss_book': s.get('logloss_book')})
    out = pd.DataFrame(rows).sort_values('rps')
    out.to_csv(os.path.join(ROOT, 'data', 'tuning2.csv'), index=False, encoding='utf-8-sig')
    print(out.head(20).round(4).to_string(index=False))
    print('\nсредний RPS по параметрам:')
    for k in keys:
        print(out.groupby(k).rps.min().round(4).to_string(), '\n')
