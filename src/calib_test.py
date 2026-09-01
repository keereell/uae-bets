# -*- coding: utf-8 -*-
"""
Улучшает ли калибровка тура точность вне выборки?

Берём walk-forward прогнозы по РПЛ и по ОАЭ. Для каждого игрового дня
подбираем dmu/dgamma так, чтобы в среднем по дню модель совпадала с рынком,
применяем и сравниваем RPS/log-loss до и после.

Ключевая честность: калибровка использует ТОЛЬКО коэффициенты (они известны
до матча), никакой информации о результате.
"""
import sys, os
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from markets import DEVIG, wdl
from model import score_matrix
from backtest import rps
from scipy.optimize import brentq
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def market_lams(q, tot_guess=2.8):
    """Грубая оценка (lh, la) из вероятностей 1X2: подбираем перекос при заданном тотале."""
    lo = np.log(q[0] / max(q[2], 1e-9))
    def f(d):
        lh = tot_guess * np.exp(d) / (1 + np.exp(d))
        la = tot_guess - lh
        p = wdl(score_matrix(lh, la, 0.0))
        return np.log(max(p['H'],1e-9) / max(p['A'],1e-9)) - lo
    try:
        d = brentq(f, -4, 4)
    except Exception:
        d = 0.0
    lh = tot_guess * np.exp(d) / (1 + np.exp(d))
    return lh, tot_guess - lh


def evaluate(path, cols, label, group='date'):
    R = pd.read_csv(path).dropna(subset=cols)
    if len(R) < 60:
        print(f'{label}: мало данных'); return
    rows = []
    for day, g in R.groupby(group):
        if len(g) < 3:
            continue
        ml, mk = [], []
        for _, r in g.iterrows():
            q = DEVIG['power'](list(r[cols].values))
            tot = r.lh + r.la     # тотал берём из модели: рынок 1X2 его не задаёт
            mk.append(market_lams(q, tot))
            ml.append((r.lh, r.la))
        ml = np.array(ml); mk = np.array(mk)
        dmu = float(np.mean(np.log(mk[:,1]) - np.log(ml[:,1])))
        dgam = float(np.mean(np.log(mk[:,0]) - np.log(ml[:,0]))) - dmu
        for (_, r), (lh, la) in zip(g.iterrows(), ml):
            lh2, la2 = np.exp(np.log(lh)+dmu+dgam), np.exp(np.log(la)+dmu)
            p0 = np.array([r.pH, r.pD, r.pA])
            pc = wdl(score_matrix(float(lh2), float(la2), 0.0))
            p1 = np.array([pc['H'], pc['D'], pc['A']])
            o = int(r.out)
            rows.append(dict(rps0=rps(p0,o), rps1=rps(p1,o),
                             ll0=-np.log(max(p0[o],1e-12)), ll1=-np.log(max(p1[o],1e-12))))
    D = pd.DataFrame(rows)
    n = len(D)
    d_rps = D.rps0.mean() - D.rps1.mean()
    se = (D.rps0 - D.rps1).std(ddof=1)/np.sqrt(n)
    print(f'{label} (N={n}, дней {R[group].nunique()}):')
    print(f'   RPS      до {D.rps0.mean():.4f}  после {D.rps1.mean():.4f}  '
          f'улучшение {d_rps:+.4f} ± {1.96*se:.4f}  t={d_rps/se:+.2f}')
    print(f'   log-loss до {D.ll0.mean():.4f}  после {D.ll1.mean():.4f}  '
          f'улучшение {D.ll0.mean()-D.ll1.mean():+.4f}')


evaluate(os.path.join(ROOT,'data','rpl','walkforward.csv'), ['AvgCH','AvgCD','AvgCA'], 'РПЛ')
print()
evaluate(os.path.join(ROOT,'data','walkforward.csv'), ['oH','oD','oA'], 'ОАЭ')
