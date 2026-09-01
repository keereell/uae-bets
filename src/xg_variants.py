# -*- coding: utf-8 -*-
"""
Какая версия xG лучше предсказывает: сырой, без пенальти, только с игры,
только при равном счёте, xGOT?
Сравнение по walk-forward RPS и log-loss на одной и той же выборке.
"""
import sys, os
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from backtest import walk_forward, summarize
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

df = pd.read_csv(os.path.join(ROOT, 'data', 'matches.csv'))
base = dict(half_life=150.0, alpha=0.30, w_goals=0.0, promoted_shift=0.25)

VARIANTS = [
    ('сырой xG (из статистики матча)', ('h_xg', 'a_xg')),
    ('сырой xG (сумма ударов)',        ('h_xg_all', 'a_xg_all')),
    ('NPxG — без пенальти',            ('h_npxg', 'a_npxg')),
    ('xG только с игры',               ('h_xg_open', 'a_xg_open')),
    ('xG при равном счёте',            ('h_xg_level', 'a_xg_level')),
    ('xGOT (в створ)',                 ('h_xgot', 'a_xgot')),
    ('только голы (w_goals=1)',        ('h_xg', 'a_xg')),
]
rows = []
for name, cols in VARIANTS:
    p = dict(base)
    p['xg_cols'] = cols
    if name.startswith('только голы'):
        p['w_goals'] = 1.0
    if not all(c in df.columns for c in cols):
        print('нет колонок', cols); continue
    r = walk_forward(df, p)
    s = summarize(r)
    rows.append(dict(вариант=name, N=s['n'], RPS=round(s['rps'], 5),
                     logloss=round(s['logloss'], 5),
                     RPS_рынок=round(s.get('rps_book', np.nan), 5)))
    print(rows[-1])

R = pd.DataFrame(rows).sort_values('RPS')
print()
print(R.to_string(index=False))
best = R.iloc[0]
worst = R[R['вариант'].str.startswith('сырой xG (из')].iloc[0]
print(f"\nлучший вариант «{best['вариант']}»: RPS {best['RPS']:.5f} "
      f"против {worst['RPS']:.5f} у сырого xG -> улучшение {worst['RPS']-best['RPS']:+.5f}")
R.to_csv(os.path.join(ROOT, 'data', 'xg_variants.csv'), index=False, encoding='utf-8-sig')
