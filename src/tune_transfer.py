# -*- coding: utf-8 -*-
"""
Переносится ли подбор гиперпараметров с одного отрезка на другой?

Считаем RPS каждой конфигурации на ДВУХ непересекающихся окнах и смотрим:
  * коррелируют ли рейтинги конфигураций между окнами
  * лучше ли выбранная по первому окну конфигурация, чем средняя, на втором
  * какой вариант xG выигрывает на каждом окне

Если корреляция около нуля, подбор гиперпараметров здесь бесполезен,
и надо брать значения из теории, а не из сетки.
"""
import sys, os, itertools
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from backtest import walk_forward, summarize
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
pd.set_option('display.width', 220)

df = pd.read_csv(os.path.join(ROOT,'data','matches.csv'))
A = ('2025-02-01', '2025-12-31')
B = ('2026-01-01', None)

grid = dict(half_life=[110,150,220], alpha=[0.15,0.30,0.60],
            w_goals=[0.0,0.2], promoted_shift=[0.0,0.25])
XG = [('h_xg_open','a_xg_open'), ('h_npxg','a_npxg'), ('h_xg','a_xg')]

rows=[]
for combo in itertools.product(*grid.values()):
    for xg in XG:
        p = dict(zip(grid.keys(), combo)); p['xg_cols']=xg
        sa = summarize(walk_forward(df, p, start_date=A[0], end_date=A[1]))
        sb = summarize(walk_forward(df, p, start_date=B[0]))
        rows.append({**{k:v for k,v in p.items() if k!='xg_cols'},
                     'xg': xg[0].replace('h_',''),
                     'rps_A': sa['rps'], 'rps_B': sb['rps'],
                     'book_B': sb.get('rps_book', np.nan)})
T = pd.DataFrame(rows)
T.to_csv(os.path.join(ROOT,'data','tune_transfer.csv'), index=False, encoding='utf-8-sig')

r = np.corrcoef(T.rps_A, T.rps_B)[0,1]
sp = T.rps_A.rank().corr(T.rps_B.rank(), method='spearman')
print(f'конфигураций: {len(T)}')
print(f'корреляция RPS между окнами: Пирсон {r:+.3f}, Спирмен {sp:+.3f}')
print()
best_A = T.sort_values('rps_A').iloc[0]
print('лучшая на окне A:', dict(best_A[['half_life','alpha','w_goals','promoted_shift','xg']]),
      f'-> rps_A {best_A.rps_A:.5f}, rps_B {best_A.rps_B:.5f}')
best_B = T.sort_values('rps_B').iloc[0]
print('лучшая на окне B:', dict(best_B[['half_life','alpha','w_goals','promoted_shift','xg']]),
      f'-> rps_A {best_B.rps_A:.5f}, rps_B {best_B.rps_B:.5f}')
rank = int((T.rps_B < best_A.rps_B).sum()) + 1
print(f'\nвыбранная по A конфигурация занимает на B место {rank} из {len(T)}')
print(f'её RPS на B: {best_A.rps_B:.5f}, медиана по всем конфигурациям на B: {T.rps_B.median():.5f}')
print(f'выигрыш от подбора: {T.rps_B.median()-best_A.rps_B:+.5f}')
print(f'линия Bet365 на B: {T.book_B.iloc[0]:.5f}')
print()
print('средний RPS по варианту xG:')
print(T.groupby('xg')[['rps_A','rps_B']].mean().round(5).to_string())
