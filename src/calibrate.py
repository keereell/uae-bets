# -*- coding: utf-8 -*-
"""
Пост-калибровка модели на результатах walk-forward.

На малой выборке (~120 эффективных матчей) оценки силы команд шумные,
поэтому модель СЛИШКОМ уверена в фаворитах. Лечится сжатием:

    s = log(lh) + log(la)          общий уровень результативности
    d = log(lh) - log(la)          перекос в пользу хозяев

    s' = s_mean + k_s * (s - s_mean) + c
    d' =                k_d * d

    lh' = exp((s' + d')/2),  la' = exp((s' - d')/2)

Параметры k_s, k_d, c подбираются по out-of-sample правдоподобию
Диксона-Коулза на реальных счетах. k_d < 1 означает переуверенность.
"""
import numpy as np
from scipy.optimize import minimize
from scipy.stats import poisson

from model import score_matrix


def apply_shrink(lh, la, k_s, k_d, c, s_mean):
    s = np.log(lh) + np.log(la)
    d = np.log(lh) - np.log(la)
    s2 = s_mean + k_s * (s - s_mean) + c
    d2 = k_d * d
    return np.exp((s2 + d2) / 2), np.exp((s2 - d2) / 2)


def fit_shrink(lh, la, hg, ag, rho=0.0):
    lh, la = np.asarray(lh, float), np.asarray(la, float)
    hg, ag = np.asarray(hg, int), np.asarray(ag, int)
    s_mean = float((np.log(lh) + np.log(la)).mean())

    def nll(th):
        k_s, k_d, c = th
        a, b = apply_shrink(lh, la, k_s, k_d, c, s_mean)
        ll = poisson.logpmf(hg, a) + poisson.logpmf(ag, b)
        if rho:
            t = np.ones(len(a))
            m = (hg == 0) & (ag == 0); t[m] = 1 - a[m] * b[m] * rho
            m = (hg == 0) & (ag == 1); t[m] = 1 + a[m] * rho
            m = (hg == 1) & (ag == 0); t[m] = 1 + b[m] * rho
            m = (hg == 1) & (ag == 1); t[m] = 1 - rho
            if (t <= 0).any():
                return 1e9
            ll = ll + np.log(t)
        return -float(ll.sum())

    r = minimize(nll, [1.0, 1.0, 0.0], method='Nelder-Mead',
                 options=dict(maxiter=2000, xatol=1e-6, fatol=1e-8))
    k_s, k_d, c = r.x
    return dict(k_s=float(k_s), k_d=float(k_d), c=float(c), s_mean=s_mean)


def calibrated_matrix(lh, la, cal, rho):
    a, b = apply_shrink(np.array([lh]), np.array([la]),
                        cal['k_s'], cal['k_d'], cal['c'], cal['s_mean'])
    return score_matrix(float(a[0]), float(b[0]), rho), float(a[0]), float(b[0])
