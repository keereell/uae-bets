# -*- coding: utf-8 -*-
"""
Системный сдвиг модели, измеренный БЕЗ коэффициентов — по фактическим голам
на walk-forward прогнозах. Если модель стабильно недооценивает
результативность или преимущество поля, это видно прямо на результатах.
"""
import sys, os
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from backtest import walk_forward
from predict import best_params
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

df = pd.read_csv(os.path.join(ROOT, 'data', 'matches.csv'))
wf = walk_forward(df, best_params())
print(f'матчей вне выборки: {len(wf)}')

# сдвиг по уровню и по полю, оценённый на фактических голах
dmu_h  = float(np.log(wf.ag.sum() / wf.la.sum()))
dgam_h = float(np.log(wf.hg.sum() / wf.lh.sum())) - dmu_h
print(f'\nПО ВСЕЙ ИСТОРИИ ({len(wf)} матчей):')
print(f'  голы хозяев : факт {wf.hg.mean():.3f}  модель {wf.lh.mean():.3f}  '
      f'-> {100*(wf.hg.mean()/wf.lh.mean()-1):+.1f}%')
print(f'  голы гостей : факт {wf.ag.mean():.3f}  модель {wf.la.mean():.3f}  '
      f'-> {100*(wf.ag.mean()/wf.la.mean()-1):+.1f}%')
print(f'  тотал       : факт {(wf.hg+wf.ag).mean():.3f}  модель {(wf.lh+wf.la).mean():.3f}  '
      f'-> {(wf.hg+wf.ag).mean()-(wf.lh+wf.la).mean():+.3f} гола')
print(f'  поправки: dmu = {dmu_h:+.4f}, dgamma = {dgam_h:+.4f}')

# стандартная ошибка поправки по уровню
n = len(wf)
se = float(np.std(np.log((wf.hg+wf.ag+0.5)/(wf.lh+wf.la)), ddof=1)/np.sqrt(n))
print(f'  ст.ошибка dmu ≈ {se:.4f} -> 95% ДИ [{dmu_h-1.96*se:+.4f}, {dmu_h+1.96*se:+.4f}]')

# то же по последнему сезону отдельно -- не растёт ли результативность
print('\nПО СЕЗОНАМ:')
wf['season'] = wf.season if 'season' in wf else np.nan
for s, g in wf.groupby('season'):
    print(f'  сезон {int(s)}: N={len(g):3d}  факт {(g.hg+g.ag).mean():.2f}  '
          f'модель {(g.lh+g.la).mean():.2f}  сдвиг {(g.hg+g.ag).mean()-(g.lh+g.la).mean():+.3f}')

# фактическая результативность лиги по сезонам
p = df[df.played]
print('\nРЕЗУЛЬТАТИВНОСТЬ ЛИГИ:')
for s, g in p.groupby('season'):
    print(f'  сезон {int(s)}: N={len(g):3d}  голов за матч {(g.hg+g.ag).mean():.2f}')
