# -*- coding: utf-8 -*-
"""
exp_rating_ensemble.py
======================
Does an ENSEMBLE of the xG Dixon-Coles model with an independent rating system
beat the Dixon-Coles model alone?

Components
----------
a) ELO       goal-difference-weighted update (Hvattum & Arntzen 2010 style),
             K and home-field advantage tuned on window A only.
b) PI        pi-ratings (Constantinou & Fenton 2013): separate home/away rating
             per team, updated from the error in GOAL DIFFERENCE via
             psi(e) = c * log10(1 + e), learning rates lambda and gamma.
c) PI_XG     same recursion but driven by the xG difference (xg_open, rescaled
             to the goal scale with an expanding-window factor).
d) COMBO     all three rating features in one regularised multinomial logit,
             refitted walk-forward. (The literature's gradient-boosted-tree
             result rests on datasets of 10^4-10^5 matches; here the walk-forward
             training set is 200-350 matches, so a boosted-tree stack is not
             defensible and a penalised linear combiner is the right-sized
             stand-in. sklearn is also not installed in this environment.)
e) ENSEMBLE  log opinion pool  p ~ p_dc^w * p_rating^(1-w),
             w chosen on window A, REPORTED on window B.

Every rating is a strictly sequential recursion over matches ordered by kickoff
timestamp: the state used to predict match t contains only matches < t.
Ratings are mapped to 1X2 with an ORDERED LOGIT that is refitted for every
evaluation day on matches strictly before that day (walk-forward, no look-ahead).

Methodology
-----------
* Windows A (2025-02-01..2025-12-31) and B (2026-01-01..) are always reported
  separately.
* All hyperparameters (K, hfa, lambda, gamma, ordered-logit half-life, and the
  pool weight w) are selected on window A ONLY. Window A numbers for the tuned
  systems are therefore IN-SAMPLE and optimistic; window B is the honest test.
* Paired differences vs the Dixon-Coles baseline are reported as
  d = rps(mine) - rps(baseline) per match, with mean, 1.96*se and t.
* Evaluation is restricted to exactly the matches the baseline walk_forward
  produced, so every comparison is paired on the same matches.

The hyperparameter grids below were narrowed after a wider boundary-check pass
(K<=60, hfa<=160, lambda<=0.35, gamma<=1.0) that was scored on WINDOW A ONLY.

Run:  python src/exp_rating_ensemble.py
"""
import os
import sys
import itertools
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from backtest import walk_forward, rps  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE = os.path.join(ROOT, 'work', 'exp_rating_ensemble_baseline.csv')

BASE_PARAMS = dict(half_life=180, alpha=0.30, w_goals=0.0,
                   promoted_shift=0.25,
                   xg_cols=('h_xg_open', 'a_xg_open'), dispersion=None)

START = '2025-02-01'
A_END = '2025-12-31'
B_START = '2026-01-01'

RNG_FLOOR = 1e-4


# --------------------------------------------------------------- ordered logit
def _ologit_nll(th, x, y, w, ridge):
    b, c1, ld = th
    c2 = c1 + np.exp(np.clip(ld, -12, 4))
    z = b * x
    F1 = expit(c1 - z)
    F2 = expit(c2 - z)
    pA = F1
    pD = np.clip(F2 - F1, 1e-12, None)
    pH = np.clip(1.0 - F2, 1e-12, None)
    p = np.where(y == 0, pH, np.where(y == 1, pD, np.clip(pA, 1e-12, None)))
    return -float(np.sum(w * np.log(p))) + ridge * b * b


class OLogit:
    """3-parameter ordered logit on a single rating feature. y: 0=H, 1=D, 2=A."""

    def __init__(self, ridge=1e-3):
        self.ridge = ridge

    def fit(self, x, y, w=None):
        x = np.asarray(x, float)
        y = np.asarray(y, int)
        w = np.ones(len(x)) if w is None else np.asarray(w, float)
        self.mu_ = float(x.mean())
        self.sd_ = float(x.std()) or 1.0
        xs = (x - self.mu_) / self.sd_
        r = minimize(_ologit_nll, np.array([1.0, -0.5, np.log(1.0)]),
                     args=(xs, y, w, self.ridge), method='BFGS',
                     options=dict(maxiter=300, gtol=1e-7))
        self.th_ = r.x
        return self

    def predict(self, x):
        x = np.atleast_1d(np.asarray(x, float))
        xs = (x - self.mu_) / self.sd_
        b, c1, ld = self.th_
        c2 = c1 + np.exp(np.clip(ld, -12, 4))
        z = b * xs
        F1 = expit(c1 - z)
        F2 = expit(c2 - z)
        p = np.column_stack([1.0 - F2, F2 - F1, F1])
        p = np.clip(p, RNG_FLOOR, None)
        return p / p.sum(axis=1, keepdims=True)


# --------------------------------------------------------------- rating engines
def elo_features(played, K=20.0, hfa=60.0, start=1500.0):
    """Goal-difference-weighted Elo. Returns per-match pre-kickoff rating diff."""
    R = {}
    out = np.empty(len(played))
    hg = played.hg.values.astype(float)
    ag = played.ag.values.astype(float)
    for i, (h, a) in enumerate(zip(played.home.values, played.away.values)):
        rh = R.setdefault(h, start)
        ra = R.setdefault(a, start)
        d = rh + hfa - ra
        out[i] = d
        e = 1.0 / (1.0 + 10.0 ** (-d / 400.0))
        gd = hg[i] - ag[i]
        s = 1.0 if gd > 0 else (0.5 if gd == 0 else 0.0)
        m = abs(gd)
        g = 1.0 if m <= 1 else (1.5 if m == 2 else (11.0 + m) / 8.0)
        upd = K * g * (s - e)
        R[h] = rh + upd
        R[a] = ra - upd
    return out


def _psi(x, c=3.0, b=3.0):
    """C&F mapping between rating scale and goal-difference scale."""
    return np.sign(x) * (10.0 ** (abs(x) / b) - 1.0)


def pi_features(played, lam=0.035, gam=0.7, c=3.0, b=3.0, target='goals'):
    """
    pi-ratings, Constantinou & Fenton (2013).

    Each team k carries R_kH (home form) and R_kA (away form).
    For match (H vs A):
        ghat_H = psi(R_HH)          expected GD of the home team when at home
        ghat_A = psi(R_AA)          expected GD of the away team when away
        ghat   = (ghat_H - ghat_A) / 2      predicted GD of THIS match
        e      = |g_obs - ghat|
        psi_e  = c * log10(1 + e)
        R_HH  += psi_e * lam * sign(g_obs - ghat)
        R_HA  += (delta R_HH) * gam
        R_AA  += psi_e * lam * sign(ghat - g_obs)
        R_AH  += (delta R_AA) * gam
    Returns the pre-kickoff predicted goal difference ghat for every match.

    target='xg' drives the update with the xG difference instead, rescaled to
    the goal scale with an EXPANDING-WINDOW factor (prior matches only).
    """
    RH, RA = {}, {}
    out = np.empty(len(played))
    hg = played.hg.values.astype(float)
    ag = played.ag.values.astype(float)
    if target == 'xg':
        hx = played.h_xg_open.values.astype(float)
        ax = played.a_xg_open.values.astype(float)
        cum_g, cum_x = 0.0, 0.0
    for i, (h, a) in enumerate(zip(played.home.values, played.away.values)):
        rhh = RH.setdefault(h, 0.0)
        raa = RA.setdefault(a, 0.0)
        RA.setdefault(h, 0.0)
        RH.setdefault(a, 0.0)
        ghat = (_psi(rhh, c, b) - _psi(raa, c, b)) / 2.0
        out[i] = ghat

        if target == 'xg':
            s = (cum_g / cum_x) if cum_x > 5.0 else 1.2
            if np.isnan(hx[i]) or np.isnan(ax[i]):
                gobs = hg[i] - ag[i]
            else:
                gobs = (hx[i] - ax[i]) * s
                cum_g += hg[i] + ag[i]
                cum_x += hx[i] + ax[i]
        else:
            gobs = hg[i] - ag[i]

        err = abs(gobs - ghat)
        pe = c * np.log10(1.0 + err)
        sgn = 1.0 if gobs > ghat else (-1.0 if gobs < ghat else 0.0)
        dH = pe * lam * sgn
        dA = -dH
        RH[h] = rhh + dH
        RA[h] = RA[h] + dH * gam
        RA[a] = raa + dA
        RH[a] = RH[a] + dA * gam
    return out


# --------------------------------------------------------------- softmax combiner
def _softmax_nll(th, X, Y, w, l2, k):
    B = th.reshape(k + 1, 3)
    Z = X @ B
    Z = Z - Z.max(axis=1, keepdims=True)
    E = np.exp(Z)
    P = E / E.sum(axis=1, keepdims=True)
    nll = -float(np.sum(w * np.log(np.clip((P * Y).sum(axis=1), 1e-12, None))))
    nll += l2 * float(np.sum(B[1:] ** 2))
    G = X.T @ (w[:, None] * (P - Y))
    G[1:] += 2 * l2 * B[1:]
    return nll, G.ravel()


class SoftmaxReg:
    """Multinomial logit over several rating features. Classes 0=H, 1=D, 2=A."""

    def __init__(self, l2=1.0):
        self.l2 = l2

    def fit(self, X, y, w=None):
        X = np.atleast_2d(np.asarray(X, float))
        n, k = X.shape
        self.mu_ = X.mean(axis=0)
        self.sd_ = np.where(X.std(axis=0) > 0, X.std(axis=0), 1.0)
        Xa = np.column_stack([np.ones(n), (X - self.mu_) / self.sd_])
        Y = np.zeros((n, 3))
        Y[np.arange(n), np.asarray(y, int)] = 1
        w = np.ones(n) if w is None else np.asarray(w, float)
        r = minimize(_softmax_nll, np.zeros((k + 1) * 3), jac=True,
                     args=(Xa, Y, w, self.l2, k), method='L-BFGS-B',
                     options=dict(maxiter=500))
        self.B_ = r.x.reshape(k + 1, 3)
        return self

    def predict(self, X):
        X = np.atleast_2d(np.asarray(X, float))
        Xa = np.column_stack([np.ones(len(X)), (X - self.mu_) / self.sd_])
        Z = Xa @ self.B_
        Z = Z - Z.max(axis=1, keepdims=True)
        E = np.exp(Z)
        P = E / E.sum(axis=1, keepdims=True)
        P = np.clip(P, RNG_FLOOR, None)
        return P / P.sum(axis=1, keepdims=True)


# --------------------------------------------------------------- walk-forward map
def rating_probs(played, feat, eval_keys, ol_half_life=None, ridge=1e-3):
    """
    Walk-forward ordered-logit mapping of a rating feature to 1X2.
    For each evaluation day, fit on matches strictly before that day.
    eval_keys: DataFrame with columns date/home/away identifying the matches to score.
    """
    d = played[['date', 'home', 'away', 'hg', 'ag', 'ts']].copy()
    d['x'] = feat
    d['y'] = np.where(d.hg > d.ag, 0, np.where(d.hg == d.ag, 1, 2))
    key = eval_keys.set_index(['date', 'home', 'away']).index
    d['_k'] = list(zip(d.date, d.home, d.away))
    want = set(key)
    res = {}
    for day in sorted({k[0] for k in want}):
        tr = d[d.date < day]
        if len(tr) < 60:
            continue
        if ol_half_life:
            ref = d.loc[d.date == day, 'ts'].min()
            w = np.exp(-np.log(2) / ol_half_life *
                       np.maximum((ref - tr.ts.values) / 86400.0, 0.0))
        else:
            w = None
        try:
            ol = OLogit(ridge).fit(tr.x.values, tr.y.values, w)
        except Exception:
            continue
        te = d[d.date == day]
        P = ol.predict(te.x.values)
        for k, p in zip(te._k, P):
            if k in want:
                res[k] = p
    idx = list(eval_keys.itertuples(index=False))
    out = np.full((len(eval_keys), 3), np.nan)
    for i, r in enumerate(idx):
        p = res.get((r.date, r.home, r.away))
        if p is not None:
            out[i] = p
    return out


def combo_probs(played, feats, eval_keys, ol_half_life=None, l2=1.0):
    """Walk-forward multinomial logit over a stack of rating features."""
    d = played[['date', 'home', 'away', 'hg', 'ag', 'ts']].copy()
    X = np.column_stack(feats)
    d['y'] = np.where(d.hg > d.ag, 0, np.where(d.hg == d.ag, 1, 2))
    d['_k'] = list(zip(d.date, d.home, d.away))
    want = set(eval_keys.set_index(['date', 'home', 'away']).index)
    res = {}
    dates = d.date.values
    for day in sorted({k[0] for k in want}):
        tr = dates < day
        if tr.sum() < 60:
            continue
        if ol_half_life:
            ref = d.loc[d.date == day, 'ts'].min()
            w = np.exp(-np.log(2) / ol_half_life *
                       np.maximum((ref - d.ts.values[tr]) / 86400.0, 0.0))
        else:
            w = None
        try:
            mdl = SoftmaxReg(l2).fit(X[tr], d.y.values[tr], w)
        except Exception:
            continue
        te = dates == day
        P = mdl.predict(X[te])
        for kk, p in zip(np.asarray(d._k.values, dtype=object)[te], P):
            if kk in want:
                res[kk] = p
    out = np.full((len(eval_keys), 3), np.nan)
    for i, r in enumerate(eval_keys.itertuples(index=False)):
        p = res.get((r.date, r.home, r.away))
        if p is not None:
            out[i] = p
    return out


# --------------------------------------------------------------- scoring helpers
def rps_vec(P, y):
    cp = np.cumsum(P[:, :2], axis=1)
    obs = np.zeros((len(y), 3))
    obs[np.arange(len(y)), y] = 1
    co = np.cumsum(obs[:, :2], axis=1)
    return ((cp - co) ** 2).sum(axis=1) / 2.0


def paired(d):
    d = np.asarray(d, float)
    d = np.where(np.abs(d) < 1e-9, 0.0, d)   # kill float noise from renormalisation
    n = len(d)
    m = d.mean()
    se = d.std(ddof=1) / np.sqrt(n)
    return dict(n=n, mean=m, ci95=1.96 * se, t=(m / se if se > 0 else np.nan))


def fmt_row(label, rps_a, dA, rps_b, dB):
    def f(x, k):
        return '' if x is None else f'{x[k]:+.5f}'
    return (f'{label:<26} {rps_a:.5f}  {f(dA,"mean"):>9} {f(dA,"ci95"):>9} '
            f'{(dA["t"] if dA else float("nan")):>6.2f}   '
            f'{rps_b:.5f}  {f(dB,"mean"):>9} {f(dB,"ci95"):>9} '
            f'{(dB["t"] if dB else float("nan")):>6.2f}')


# --------------------------------------------------------------- main
def main():
    df = pd.read_csv(os.path.join(ROOT, 'data', 'matches.csv'))

    # ---- baseline Dixon-Coles, one walk_forward over the union of both windows
    if os.path.exists(CACHE):
        base = pd.read_csv(CACHE)
    else:
        base = walk_forward(df, BASE_PARAMS, start_date=START, end_date=None)
        os.makedirs(os.path.dirname(CACHE), exist_ok=True)
        base.to_csv(CACHE, index=False)
    base = base.sort_values(['date', 'home', 'away']).reset_index(drop=True)

    inA = (base.date >= START) & (base.date <= A_END)
    inB = base.date >= B_START
    y = base.out.values.astype(int)
    P_dc = base[['pH', 'pD', 'pA']].values
    r_dc = base.rps.values

    print('=' * 108)
    print('BASELINE Dixon-Coles (xG, half_life=180, alpha=0.30, xg_open)')
    print(f'  window A  n={inA.sum():3d}  RPS={r_dc[inA.values].mean():.5f}')
    print(f'  window B  n={inB.sum():3d}  RPS={r_dc[inB.values].mean():.5f}')
    if 'rps_book' in base:
        bk = base.dropna(subset=['rps_book'])
        ba = bk[(bk.date >= START) & (bk.date <= A_END)]
        bb = bk[bk.date >= B_START]
        print(f'  Bet365    A n={len(ba)} RPS={ba.rps_book.mean():.5f} | '
              f'B n={len(bb)} RPS={bb.rps_book.mean():.5f}')
    print('=' * 108)

    played = df[df.played & df.hg.notna()].sort_values('ts').reset_index(drop=True)
    keys = base[['date', 'home', 'away']].copy()

    # ---------------------------------------------------------------- tuning on A
    def score_on_A(P):
        ok = inA.values & ~np.isnan(P[:, 0])
        return rps_vec(P[ok], y[ok]).mean(), ok.sum()

    cache = {}

    def get_probs(kind, hp):
        k = (kind, hp)
        if k in cache:
            return cache[k]
        if kind == 'elo':
            K, hfa, hl = hp
            f = elo_features(played, K=K, hfa=hfa)
        elif kind == 'pi':
            lam, gam, hl = hp
            f = pi_features(played, lam=lam, gam=gam, target='goals')
        else:
            lam, gam, hl = hp
            f = pi_features(played, lam=lam, gam=gam, target='xg')
        P = rating_probs(played, f, keys, ol_half_life=hl)
        cache[k] = P
        return P

    grids = {
        'elo': list(itertools.product([20, 25, 32, 40], [80, 120, 160, 220], [None, 365])),
        'pi': list(itertools.product([0.12, 0.18, 0.25, 0.35],
                                     [0.7, 0.9, 1.0], [None, 365])),
        'pi_xg': list(itertools.product([0.12, 0.18, 0.25, 0.35],
                                        [0.7, 0.9, 1.0], [None, 365])),
    }

    best = {}
    print('\n--- hyperparameter selection, WINDOW A ONLY ---')
    for kind, grid in grids.items():
        rows = []
        for hp in grid:
            P = get_probs(kind, hp)
            s, n = score_on_A(P)
            rows.append((s, hp, n))
        rows.sort()
        best[kind] = rows[0][1]
        print(f'{kind:6s} best={rows[0][1]}  RPS_A={rows[0][0]:.5f}   '
              f'(worst on grid {rows[-1][0]:.5f}, spread {rows[-1][0]-rows[0][0]:.5f})')
        for s, hp, n in rows[:3]:
            print(f'        {hp}  {s:.5f}')

    # published C&F defaults, for reference (not tuned)
    ref_pi = (0.035, 0.7, None)

    # ---------------------------------------------------------------- components
    # ---- combined-ratings model: all three features in one penalised logit
    f_elo = elo_features(played, K=best['elo'][0], hfa=best['elo'][1])
    f_pi = pi_features(played, lam=best['pi'][0], gam=best['pi'][1], target='goals')
    f_pix = pi_features(played, lam=best['pi_xg'][0], gam=best['pi_xg'][1], target='xg')
    combo_grid = [(l2, hl) for l2 in (0.3, 1.0, 3.0, 10.0) for hl in (None, 365)]
    crows = []
    for l2, hl in combo_grid:
        P = combo_probs(played, [f_elo, f_pi, f_pix], keys, ol_half_life=hl, l2=l2)
        crows.append((score_on_A(P)[0], (l2, hl), P))
    crows.sort(key=lambda r: r[0])
    print(f'combo  best=(l2,hl)={crows[0][1]}  RPS_A={crows[0][0]:.5f}   '
          f'(worst on grid {crows[-1][0]:.5f})')
    P_combo = crows[0][2]

    comps = {
        'ELO tuned-A': get_probs('elo', best['elo']),
        'PIgoals tuned-A': get_probs('pi', best['pi']),
        'PIgoals C&F.035/.7': get_probs('pi', ref_pi),
        'PIxG tuned-A': get_probs('pi_xg', best['pi_xg']),
        'COMBO 3 ratings': P_combo,
    }

    hdr = (f'{"model":<26} {"RPS_A":>7}  {"dA":>9} {"+-95%":>9} {"t":>6}   '
           f'{"RPS_B":>7}  {"dB":>9} {"+-95%":>9} {"t":>6}')

    def report(name, P):
        okA = inA.values & ~np.isnan(P[:, 0])
        okB = inB.values & ~np.isnan(P[:, 0])
        rA = rps_vec(P[okA], y[okA])
        rB = rps_vec(P[okB], y[okB])
        dA = paired(rA - r_dc[okA])
        dB = paired(rB - r_dc[okB])
        print(fmt_row(name, rA.mean(), dA, rB.mean(), dB))
        return rA.mean(), rB.mean(), dA, dB

    print('\n' + '=' * 108)
    print('STANDALONE COMPONENTS  (d = RPS(model) - RPS(DixonColes); negative = better)')
    print('=' * 108)
    print(hdr)
    print(fmt_row('DixonColes baseline', r_dc[inA.values].mean(), None,
                  r_dc[inB.values].mean(), None).replace('nan', '  -'))
    comp_scores = {}
    for nm, P in comps.items():
        comp_scores[nm] = report(nm, P)

    # ---------------------------------------------------------------- ensemble
    def pool(Pd, Pr, w):
        with np.errstate(divide='ignore'):
            L = w * np.log(np.clip(Pd, 1e-12, None)) + (1 - w) * np.log(np.clip(Pr, 1e-12, None))
        P = np.exp(L - L.max(axis=1, keepdims=True))
        return P / P.sum(axis=1, keepdims=True)

    ws = np.round(np.arange(0.0, 1.001, 0.05), 3)
    print('\n' + '=' * 108)
    print('LOG-OPINION-POOL ENSEMBLE   p ~ p_dc^w * p_rating^(1-w)')
    print('w chosen on WINDOW A, reported on WINDOW B')
    print('=' * 108)
    print(hdr)
    ens_rows = []
    for nm, Pr in comps.items():
        okA = inA.values & ~np.isnan(Pr[:, 0])
        curve = []
        for w in ws:
            P = pool(P_dc[okA], Pr[okA], w)
            curve.append(rps_vec(P, y[okA]).mean())
        curve = np.array(curve)
        w_star = float(ws[int(np.argmin(curve))])
        nm2 = f'ENS {nm} w={w_star:.2f}'
        Pens = np.full_like(Pr, np.nan)
        ok = ~np.isnan(Pr[:, 0])
        Pens[ok] = pool(P_dc[ok], Pr[ok], w_star)
        rA, rB, dA, dB = report(nm2, Pens)
        okB = inB.values & ~np.isnan(Pr[:, 0])
        curveB = np.array([rps_vec(pool(P_dc[okB], Pr[okB], w), y[okB]).mean() for w in ws])
        w_star_B = float(ws[int(np.argmin(curveB))])
        ens_rows.append((nm, w_star, curve, rA, rB, dA, dB, w_star_B, curveB))
        # honesty check: fixed 50/50 pool, no tuning at all
        Pfix = np.full_like(Pr, np.nan)
        Pfix[ok] = pool(P_dc[ok], Pr[ok], 0.5)
        report(f'ENS {nm} w=0.50 fix', Pfix)

    print('\n--- window-A weight curves (RPS_A by w; w=1 is pure Dixon-Coles) ---')
    print('w:      ' + ' '.join(f'{w:6.2f}' for w in ws[::2]))
    for nm, w_star, curve, *_ in ens_rows:
        print(f'A {nm:<22} ' + ' '.join(f'{c:6.4f}' for c in curve[::2]))
    print('--- same curves on window B (NOT used for selection, shown only to expose instability) ---')
    print('w:      ' + ' '.join(f'{w:6.2f}' for w in ws[::2]))
    for nm, w_star, curve, rA, rB, dA, dB, w_star_B, curveB in ens_rows:
        print(f'B {nm:<22} ' + ' '.join(f'{c:6.4f}' for c in curveB[::2]) +
              f'   w*_A={w_star:.2f}  w*_B={w_star_B:.2f}')

    # ---------------------------------------------------------------- verdict
    print('\n' + '=' * 108)
    print('VERDICT (window B = out-of-sample for every tuned choice)')
    print('=' * 108)
    print(f'  Dixon-Coles alone            RPS_B = {r_dc[inB.values].mean():.5f}')
    for nm, (rA, rB, dA, dB) in comp_scores.items():
        print(f'  {nm:<28} RPS_B = {rB:.5f}   dB = {dB["mean"]:+.5f} +- {dB["ci95"]:.5f}')
    for nm, w_star, curve, rA, rB, dA, dB, w_star_B, curveB in ens_rows:
        print(f'  ENS {nm:<24} RPS_B = {rB:.5f}   '
              f'dB = {dB["mean"]:+.5f} +- {dB["ci95"]:.5f}  (w*={w_star:.2f} from A)')
    print('\n  minimum detectable paired difference on ~250 matches ~ 0.0035')


if __name__ == '__main__':
    main()
