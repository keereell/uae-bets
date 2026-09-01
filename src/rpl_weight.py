# -*- coding: utf-8 -*-
"""Бутстрап веса модели рядом с рынком: реальный сигнал или шум выборки."""
import sys, os
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from markets import DEVIG
from scipy.optimize import minimize_scalar
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
rng = np.random.default_rng(11)

def weight(P, Q, y):
    def ll(w):
        z = w*np.log(np.clip(P,1e-9,1)) + (1-w)*np.log(np.clip(Q,1e-9,1))
        z -= z.max(axis=1, keepdims=True)
        e = np.exp(z); pb = e/e.sum(axis=1, keepdims=True)
        return -np.mean(np.log(np.clip(pb[np.arange(len(y)), y], 1e-12, 1)))
    return float(minimize_scalar(ll, bounds=(-1.0, 2.0), method='bounded').x)

for name, path, cols in (
    ('РПЛ', os.path.join(ROOT,'data','rpl','walkforward.csv'), ['AvgCH','AvgCD','AvgCA']),
    ('ОАЭ', os.path.join(ROOT,'data','walkforward.csv'), ['oH','oD','oA'])):
    R = pd.read_csv(path).dropna(subset=cols)
    P = R[['pH','pD','pA']].values
    Q = np.array([DEVIG['power'](list(r)) for r in R[cols].values])
    y = R.out.values.astype(int)
    w0 = weight(P,Q,y)
    bs = []
    for _ in range(1500):
        i = rng.integers(0, len(y), len(y))
        try: bs.append(weight(P[i],Q[i],y[i]))
        except Exception: pass
    bs = np.array(bs)
    lo, hi = np.percentile(bs,[2.5,97.5])
    print(f'{name}: N={len(y):4d}  вес модели {w0:+.3f}  95% ДИ [{lo:+.2f}, {hi:+.2f}]  '
          f'P(вес>0) = {(bs>0).mean():.1%}')
