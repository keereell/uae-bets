# -*- coding: utf-8 -*-
"""Есть ли у модели СИСТЕМНОЕ смещение относительно линии по туру."""
import sys, os
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model import DixonColes, detect_newcomers, score_matrix
from markets import wdl
from backtest import walk_forward
from calibrate import fit_shrink, apply_shrink
from parse_betcity import load_all
from implied import fit_implied
from predict import best_params
from teams import to_en
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

df = pd.read_csv(os.path.join(ROOT,'data','matches.csv'))
params = best_params()
wf = walk_forward(df, params)
cal = fit_shrink(wf.lh, wf.la, wf.hg, wf.ag)
cal['k_s']=float(np.clip(cal['k_s'],0,1.5)); cal['k_d']=float(np.clip(cal['k_d'],0.3,1.5))
nc = detect_newcomers(df); up = df[~df.played].sort_values('ts')
m = DixonColes(**params).fit(df, ref_ts=float(up.ts.min()), newcomers=nc[max(nc)])

rows=[]
for head, brows in load_all():
    h,a = to_en(head.get('home')), to_en(head.get('away'))
    if h not in m.idx or a not in m.idx: continue
    imp = fit_implied(brows, head.get('main',{}))
    if imp is None: continue
    lh0,la0 = m.lambdas(h,a)
    x,y = apply_shrink(np.array([lh0]),np.array([la0]),cal['k_s'],cal['k_d'],cal['c'],cal['s_mean'])
    lh,la = float(x[0]),float(y[0])
    pm = wdl(score_matrix(lh,la,m.rho)); pi = wdl(imp['M'])
    rows.append(dict(матч=f"{head['home'][:18]} — {head['away'][:16]}",
        мод_тотал=lh+la, рын_тотал=imp['lh']+imp['la'],
        мод_λ1=lh, рын_λ1=imp['lh'], мод_λ2=la, рын_λ2=imp['la'],
        мод_PH=pm['H'], рын_PH=pi['H'], мод_PD=pm['D'], рын_PD=pi['D']))
R = pd.DataFrame(rows)
pd.set_option('display.width',200)
print(R.round(3).to_string(index=False))
print()
print(f"СРЕДНЕЕ ПО ТУРУ:")
print(f"  тотал:            модель {R.мод_тотал.mean():.3f}  рынок {R.рын_тотал.mean():.3f}  "
      f"-> смещение {R.мод_тотал.mean()-R.рын_тотал.mean():+.3f} гола")
print(f"  λ хозяев:         модель {R.мод_λ1.mean():.3f}  рынок {R.рын_λ1.mean():.3f}  "
      f"-> {100*(np.log(R.мод_λ1).mean()-np.log(R.рын_λ1).mean()):+.1f}% в логарифме")
print(f"  λ гостей:         модель {R.мод_λ2.mean():.3f}  рынок {R.рын_λ2.mean():.3f}  "
      f"-> {100*(np.log(R.мод_λ2).mean()-np.log(R.рын_λ2).mean()):+.1f}% в логарифме")
print(f"  P(победа хозяев): модель {R.мод_PH.mean():.3f}  рынок {R.рын_PH.mean():.3f}  "
      f"-> {100*(R.мод_PH.mean()-R.рын_PH.mean()):+.1f} п.п.")
print(f"  P(ничья):         модель {R.мод_PD.mean():.3f}  рынок {R.рын_PD.mean():.3f}  "
      f"-> {100*(R.мод_PD.mean()-R.рын_PD.mean()):+.1f} п.п.")
dmu = float(np.log(R.рын_λ2).mean() - np.log(R.мод_λ2).mean())
dgam = float(np.log(R.рын_λ1).mean() - np.log(R.мод_λ1).mean()) - dmu
print(f"\nПОПРАВКИ ДЛЯ УСТРАНЕНИЯ СМЕЩЕНИЯ: dmu = {dmu:+.4f}, dgamma = {dgam:+.4f}")
