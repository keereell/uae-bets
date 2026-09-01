# -*- coding: utf-8 -*-
"""
exp_structure.py -- does a different COUNT DISTRIBUTION or a RICHER TEAM STRUCTURE
beat the current Dixon-Coles/xG model?

Variants tested (each measured on window A and window B separately, paired vs baseline):

  (a) count distribution on the goals stage:
        negative binomial (Var = phi*lambda, phi > 1) and Conway-Maxwell-Poisson
        (nu < 1 over-, nu > 1 under-dispersed), both mean-matched to the model's
        lambda so ONLY the shape changes.  Also a per-fold in-sample MLE of the
        dispersion (uses training matches only -> no look-ahead).
  (b) rho: the tau-only profile likelihood used in model.py is compared against the
        FULL weighted DC likelihood over all scores; sign is checked; and the
        constrained fit rho <= 0 and fixed rho values are evaluated.
  (c) separate home/away team strengths (atk_home/atk_away/def_home/def_away) with
        L2 shrinkage kappa toward the common value (kappa -> inf reproduces baseline).
  (d) per-team home advantage with L2 shrinkage (stage 2, on goals).
  (e) team-level "finishing" (goals minus xG persistence) added back into the
        lambdas, with shrinkage; attacking side, defensive/GK side, and both.

Nothing here is tuned on the reported windows.  Free parameters are either
(i) scanned and reported as a full profile on BOTH windows, or (ii) estimated
inside each walk-forward fold from prior matches only.

Run:  python src/exp_structure.py
"""
import os
import sys
import time

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import gammaln
from scipy.stats import nbinom, poisson

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)

from model import DixonColes, detect_newcomers, dc_tau, score_matrix, MAXG  # noqa: E402
from markets import wdl  # noqa: E402
from backtest import rps  # noqa: E402

BASE = dict(half_life=180.0, alpha=0.30, w_goals=0.0, promoted_shift=0.25,
            xg_cols=('h_xg_open', 'a_xg_open'), dispersion=None)
START = '2025-02-01'
SPLIT = '2025-12-31'          # window A: <= SPLIT ; window B: > SPLIT
RHO_GRID = np.linspace(-0.30, 0.20, 101)
K = np.arange(MAXG + 1)
LOGFACT = gammaln(K + 1.0)


# =====================================================================
#                       COUNT DISTRIBUTIONS
# =====================================================================
def cmp_pmf(lam, nu, iters=60):
    """Conway-Maxwell-Poisson pmf on 0..MAXG, MEAN-MATCHED to `lam`.

    p(x) ∝ theta^x / (x!)^nu ; theta solved so that E[X] == lam exactly.
    Vectorised over an array of lambdas.  nu == 1 gives the (truncated,
    renormalised) Poisson, i.e. it reproduces the baseline to ~1e-9.
    """
    lam = np.atleast_1d(np.asarray(lam, float))
    t = np.log(np.maximum(lam, 1e-6))          # start at the Poisson value
    for _ in range(iters):
        z = t[:, None] * K[None, :] - nu * LOGFACT[None, :]
        z -= z.max(axis=1, keepdims=True)
        p = np.exp(z)
        p /= p.sum(axis=1, keepdims=True)
        m = p @ K
        v = p @ (K ** 2) - m ** 2
        step = (lam - m) / np.maximum(v, 1e-9)
        t = t + np.clip(step, -2.0, 2.0)
        if np.max(np.abs(lam - m)) < 1e-10:
            break
    z = t[:, None] * K[None, :] - nu * LOGFACT[None, :]
    z -= z.max(axis=1, keepdims=True)
    p = np.exp(z)
    return p / p.sum(axis=1, keepdims=True)


def nb_pmf(lam, phi):
    """Negative binomial pmf on 0..MAXG with mean lam and Var = phi*lam (phi>1)."""
    lam = np.atleast_1d(np.asarray(lam, float))
    if phi <= 1.0 + 1e-9:
        p = poisson.pmf(K[None, :], lam[:, None])
    else:
        r = lam / (phi - 1.0)
        p = nbinom.pmf(K[None, :], r[:, None], (r / (r + lam))[:, None])
    return p / p.sum(axis=1, keepdims=True)


def margins(lh, la, kind, par):
    """Marginal goal pmfs for the home/away side under the chosen count law."""
    if kind == 'poisson':
        return (poisson.pmf(K, lh) / poisson.pmf(K, lh).sum(),
                poisson.pmf(K, la) / poisson.pmf(K, la).sum())
    if kind == 'cmp':
        return cmp_pmf([lh], par)[0], cmp_pmf([la], par)[0]
    if kind == 'nb':
        return nb_pmf([lh], par)[0], nb_pmf([la], par)[0]
    raise ValueError(kind)


def matrix_from(lh, la, rho, kind='poisson', par=None):
    ph, pa = margins(lh, la, kind, par)
    m = np.outer(ph, pa)
    if rho:
        m = m * dc_tau(rho, lh, la)
    return m / m.sum()


# =====================================================================
#                    GENERALISED DIXON-COLES
# =====================================================================
class ExpDC(DixonColes):
    """DixonColes + optional home/away split strengths, per-team home advantage,
    team finishing term.  Stage-1 target and all baseline mechanics unchanged."""

    def __init__(self, split_kappa=None, ha_kappa=None,
                 fin_w=0.0, fin_k0=6.0, fin_side='att',
                 fin_cols=('h_xg_all', 'a_xg_all'), **kw):
        super().__init__(**kw)
        self.split_kappa = split_kappa     # None -> no home/away split
        self.ha_kappa = ha_kappa           # None -> single shared home advantage
        self.fin_w = float(fin_w)          # 0 -> no finishing term
        self.fin_k0 = float(fin_k0)        # shrinkage strength, in xG units
        self.fin_side = fin_side           # 'att' | 'def' | 'both'
        self.fin_cols = tuple(fin_cols)
        self.rho_profile = None
        self.rho_mle = 0.0
        self.rho_neg = 0.0

    # ---------------------------------------------------------- finishing
    def _finishing(self, d, w):
        n = len(self.teams)
        c0, c1 = self.fin_cols
        if c0 not in d.columns:
            c0, c1 = self.xg_cols
        hx = d[c0].values.astype(float)
        ax = d[c1].values.astype(float)
        ok = ~(np.isnan(hx) | np.isnan(ax))
        hi = d.home.map(self.idx).values
        ai = d.away.map(self.idx).values
        hg = d.hg.values.astype(float)
        ag = d.ag.values.astype(float)
        ww = w * ok
        Gf = np.zeros(n); Xf = np.zeros(n); Ga = np.zeros(n); Xa = np.zeros(n)
        np.add.at(Gf, hi, ww * hg); np.add.at(Gf, ai, ww * ag)
        np.add.at(Xf, hi, ww * np.nan_to_num(hx)); np.add.at(Xf, ai, ww * np.nan_to_num(ax))
        np.add.at(Ga, hi, ww * ag); np.add.at(Ga, ai, ww * hg)
        np.add.at(Xa, hi, ww * np.nan_to_num(ax)); np.add.at(Xa, ai, ww * np.nan_to_num(hx))
        k0 = self.fin_k0
        f_att = np.log((Gf + k0) / (Xf + k0))
        f_def = np.log((Ga + k0) / (Xa + k0))
        self.f_att = f_att - f_att.mean()
        self.f_def = f_def - f_def.mean()
        if self.fin_side == 'att':
            self.f_def = np.zeros(n)
        elif self.fin_side == 'def':
            self.f_att = np.zeros(n)

    def _fin_adj(self, i, j):
        """(home adjustment, away adjustment) for the pair (team i home, team j away)."""
        if self.fin_w == 0.0:
            return 0.0, 0.0
        return (self.fin_w * (self.f_att[i] + self.f_def[j]),
                self.fin_w * (self.f_att[j] + self.f_def[i]))

    # ---------------------------------------------------------- fit
    def fit(self, d, ref_ts=None, newcomers=()):
        d = d[d.played & d.hg.notna()].copy()
        if self.xg_scale is None:
            c0, c1 = self.xg_cols
            m = d.dropna(subset=[c0, c1])
            self.xg_scale = (float((m.hg + m.ag).sum() / (m[c0] + m[c1]).sum())
                             if len(m) and (m[c0] + m[c1]).sum() > 0 else 1.0)
        ref_ts = ref_ts if ref_ts is not None else d.ts.max()
        dt_days = np.maximum((ref_ts - d.ts.values) / 86400.0, 0.0)
        w = np.exp(-np.log(2) / self.half_life * dt_days)

        teams = sorted(set(d.home) | set(d.away))
        self.teams = teams
        self.idx = {t: i for i, t in enumerate(teams)}
        n = len(teams)
        hi = d.home.map(self.idx).values
        ai = d.away.map(self.idx).values
        yh, ya = self._target(d)

        prior = np.zeros(2 * n)
        for t in newcomers:
            if t in self.idx:
                prior[self.idx[t]] = -self.promoted_shift
                prior[n + self.idx[t]] = -self.promoted_shift

        split = self.split_kappa is not None
        npar = 2 + (4 * n if split else 2 * n)
        kap = float(self.split_kappa) if split else 0.0
        state = dict(inv_phi=1.0)

        def unpack(th):
            mu, gam = th[0], th[1]
            atk, dfn = th[2:2 + n], th[2 + n:2 + 2 * n]
            if split:
                da, dd = th[2 + 2 * n:2 + 3 * n], th[2 + 3 * n:2 + 4 * n]
            else:
                da = dd = np.zeros(n)
            return mu, gam, atk, dfn, da, dd

        def lams(th):
            mu, gam, atk, dfn, da, dd = unpack(th)
            eh = mu + gam + atk[hi] + da[hi] - dfn[ai] + dd[ai]
            ea = mu + atk[ai] - da[ai] - dfn[hi] - dd[hi]
            return eh, ea, np.exp(np.clip(eh, -6, 3)), np.exp(np.clip(ea, -6, 3))

        def nll1(th):
            eh, ea, lh, la = lams(th)
            v = state['inv_phi'] * (np.sum(w * (lh - yh * eh)) + np.sum(w * (la - ya * ea)))
            v += self.alpha * np.sum((th[2:2 + 2 * n] - prior) ** 2)
            if split:
                v += kap * np.sum(th[2 + 2 * n:] ** 2)
            return v

        def g1(th):
            eh, ea, lh, la = lams(th)
            rh = state['inv_phi'] * w * (lh - yh)
            ra = state['inv_phi'] * w * (la - ya)
            g = np.zeros_like(th)
            g[0] = rh.sum() + ra.sum()
            g[1] = rh.sum()
            np.add.at(g, 2 + hi, rh)
            np.add.at(g, 2 + ai, ra)
            np.add.at(g, 2 + n + ai, -rh)
            np.add.at(g, 2 + n + hi, -ra)
            if split:
                np.add.at(g, 2 + 2 * n + hi, rh)
                np.add.at(g, 2 + 2 * n + ai, -ra)
                np.add.at(g, 2 + 3 * n + ai, rh)
                np.add.at(g, 2 + 3 * n + hi, -ra)
                g[2 + 2 * n:] += 2 * kap * th[2 + 2 * n:]
            g[2:2 + 2 * n] += 2 * self.alpha * (th[2:2 + 2 * n] - prior)
            return g

        th0 = np.zeros(npar)
        th0[0] = np.log(max(yh.mean(), 0.1))
        th0[1] = 0.05
        th0[2:2 + 2 * n] = prior
        r1 = minimize(nll1, th0, jac=g1, method='L-BFGS-B',
                      options=dict(maxiter=1000, ftol=1e-12))
        if self.dispersion is None:
            _, _, lh0, la0 = lams(r1.x)
            num = (np.sum((yh - lh0) ** 2 / np.maximum(lh0, 1e-6)) +
                   np.sum((ya - la0) ** 2 / np.maximum(la0, 1e-6)))
            self.dispersion = float(np.clip(num / max(2 * len(d) - npar, 1), 0.15, 2.0))
        state['inv_phi'] = 1.0 / self.dispersion
        if abs(state['inv_phi'] - 1.0) > 1e-6:
            r1 = minimize(nll1, r1.x, jac=g1, method='L-BFGS-B',
                          options=dict(maxiter=1000, ftol=1e-12))
        mu1, gam1, atk, dfn, da, dd = unpack(r1.x)
        self.atk, self.dfn, self.da, self.dd = atk, dfn, da, dd

        # ------- stage 2: level / home advantage on ACTUAL GOALS
        if self.fin_w:
            self._finishing(d, w)
        else:
            self.f_att = np.zeros(n)
            self.f_def = np.zeros(n)
        fh = self.fin_w * (self.f_att[hi] + self.f_def[ai])
        fa = self.fin_w * (self.f_att[ai] + self.f_def[hi])
        base_h = atk[hi] + da[hi] - dfn[ai] + dd[ai] + fh
        base_a = atk[ai] - da[ai] - dfn[hi] - dd[hi] + fa
        gh = d.hg.values.astype(float)
        ga = d.ag.values.astype(float)
        self._cov_mean = np.zeros(0)
        self.beta = np.zeros(0)
        per_ha = self.ha_kappa is not None
        kg = float(self.ha_kappa) if per_ha else 0.0

        def nll2(th):
            mu, gam = th[0], th[1]
            gt = th[2:] if per_ha else np.zeros(n)
            eh = mu + gam + (gt[hi] if per_ha else 0.0) + base_h
            ea = mu + base_a
            lh = np.exp(np.clip(eh, -6, 3))
            la = np.exp(np.clip(ea, -6, 3))
            v = np.sum(w * (lh - gh * eh)) + np.sum(w * (la - ga * ea))
            if per_ha:
                v += kg * np.sum(gt ** 2)
            return v

        x0 = np.concatenate([[mu1, gam1], np.zeros(n if per_ha else 0)])
        r2 = minimize(nll2, x0, method='L-BFGS-B', options=dict(maxiter=800))
        self.mu, self.gamma = r2.x[0], r2.x[1]
        self.gam_t = r2.x[2:] if per_ha else np.zeros(n)
        self.eff_n = float(w.sum())
        self.rho = self._fit_rho2(d, hi, ai, w)
        return self

    # ---------------------------------------------------------- rho
    def _fit_rho2(self, d, hi, ai, w):
        lh, la = self._lam_arrays(hi, ai)
        hg = d.hg.values.astype(int)
        ag = d.ag.values.astype(int)
        m00 = (hg == 0) & (ag == 0)
        m01 = (hg == 0) & (ag == 1)
        m10 = (hg == 1) & (ag == 0)
        m11 = (hg == 1) & (ag == 1)
        vals = []
        for rho in RHO_GRID:
            t = np.ones(len(d))
            t[m00] = 1 - lh[m00] * la[m00] * rho
            t[m01] = 1 + lh[m01] * rho
            t[m10] = 1 + la[m10] * rho
            t[m11] = 1 - rho
            vals.append(1e9 if (t <= 0).any() else -float(np.sum(w * np.log(t))))
        vals = np.asarray(vals)
        self.rho_profile = vals
        self.rho_mle = float(RHO_GRID[int(np.argmin(vals))])
        neg = RHO_GRID <= 0
        self.rho_neg = float(RHO_GRID[neg][int(np.argmin(vals[neg]))])
        return self.rho_mle

    def _lam_arrays(self, hi, ai):
        fh = self.fin_w * (self.f_att[hi] + self.f_def[ai])
        fa = self.fin_w * (self.f_att[ai] + self.f_def[hi])
        eh = (self.mu + self.gamma + self.gam_t[hi] + self.atk[hi] + self.da[hi]
              - self.dfn[ai] + self.dd[ai] + fh)
        ea = self.mu + self.atk[ai] - self.da[ai] - self.dfn[hi] - self.dd[hi] + fa
        return np.exp(eh), np.exp(ea)

    # ---------------------------------------------------------- predict
    def lambdas(self, home, away, cov=None):
        i, j = self.idx.get(home), self.idx.get(away)
        if i is None or j is None:
            raise KeyError('not in model')
        fh, fa = self._fin_adj(i, j)
        lh = np.exp(self.mu + self.gamma + self.gam_t[i] + self.atk[i] + self.da[i]
                    - self.dfn[j] + self.dd[j] + fh)
        la = np.exp(self.mu + self.atk[j] - self.da[j] - self.dfn[i] - self.dd[i] + fa)
        return float(lh), float(la)

    def matrix(self, home, away, cov=None):
        lh, la = self.lambdas(home, away)
        return score_matrix(lh, la, self.rho), lh, la


# =====================================================================
#                          WALK FORWARD
# =====================================================================
def wf(df, cls=ExpDC, params=None, start_date=START, min_train=120, extra=None):
    """Walk-forward that stores lh/la/rho per match so that any purely predictive
    variant (count law, rho rule) can be re-scored without refitting."""
    params = dict(params or {})
    nc = detect_newcomers(df)
    played = df[df.played & df.hg.notna()].sort_values('ts').reset_index(drop=True)
    days = sorted(played.loc[played.date >= start_date, 'date'].unique())
    rows = []
    for day in days:
        train = played[played.date < day]
        if len(train) < min_train:
            continue
        test = played[played.date == day]
        ref_ts = test.ts.min()
        new_here = set()
        for s in test.season.unique():
            new_here |= nc.get(s, set())
        try:
            m = cls(**params).fit(train, ref_ts=ref_ts, newcomers=new_here)
        except Exception as e:                      # pragma: no cover
            print('fit fail', day, e)
            continue
        for _, r in test.iterrows():
            if r.home not in m.idx or r.away not in m.idx:
                continue
            lh, la = m.lambdas(r.home, r.away)
            M = score_matrix(lh, la, m.rho)
            p = wdl(M)
            out = 0 if r.hg > r.ag else (1 if r.hg == r.ag else 2)
            pm = np.array([p['H'], p['D'], p['A']])
            row = dict(key=f'{r.date}|{r.home}|{r.away}', date=r.date, out=out,
                       hg=int(r.hg), ag=int(r.ag), lh=lh, la=la,
                       rho=m.rho, rho_mle=m.rho_mle, rho_neg=m.rho_neg,
                       pH=pm[0], pD=pm[1], pA=pm[2],
                       ll=-np.log(max(pm[out], 1e-12)), rps=rps(pm, out))
            if extra:
                row.update(extra(m, r))
            rows.append(row)
    return pd.DataFrame(rows)


def rescore(base, kind='poisson', par=None, rho_rule='mle'):
    """Re-score a stored walk-forward pass under a different count law / rho rule."""
    out = base.copy()
    rr, pp, ll = [], [], []
    for _, r in base.iterrows():
        if rho_rule == 'mle':
            rho = r.rho_mle
        elif rho_rule == 'neg':
            rho = r.rho_neg
        elif rho_rule == 'zero':
            rho = 0.0
        else:
            rho = float(rho_rule)
        M = matrix_from(r.lh, r.la, rho, kind, par)
        p = wdl(M)
        pm = np.array([p['H'], p['D'], p['A']])
        pp.append(pm)
        rr.append(rps(pm, int(r.out)))
        ll.append(-np.log(max(pm[int(r.out)], 1e-12)))
    pp = np.array(pp)
    out['pH'], out['pD'], out['pA'] = pp[:, 0], pp[:, 1], pp[:, 2]
    out['rps'] = rr
    out['ll'] = ll
    return out


# =====================================================================
#                          REPORTING
# =====================================================================
def paired(var, base, label):
    """Paired comparison per window.  d<0 == variant better."""
    v = var.set_index('key')
    b = base.set_index('key')
    keys = b.index.intersection(v.index)
    rows = []
    for wname, mask in (('A', b.loc[keys, 'date'] <= SPLIT),
                        ('B', b.loc[keys, 'date'] > SPLIT),
                        ('both', pd.Series(True, index=keys))):
        k = keys[mask.values]
        d = (v.loc[k, 'rps'] - b.loc[k, 'rps']).values
        n = len(d)
        se = d.std(ddof=1) / np.sqrt(n) if n > 1 else np.nan
        rows.append(dict(variant=label, win=wname, n=n,
                         rps=v.loc[k, 'rps'].mean(), rps_base=b.loc[k, 'rps'].mean(),
                         d=d.mean(), ci95=1.96 * se, t=(d.mean() / se if se else np.nan),
                         ll=v.loc[k, 'll'].mean(),
                         pD=v.loc[k, 'pD'].mean(), pD_base=b.loc[k, 'pD'].mean()))
    return rows


def show(rows):
    t = pd.DataFrame(rows)
    lines = []
    for var in t.variant.unique():
        s = t[t.variant == var].set_index('win')
        lines.append('{:<34s} A {:.5f} d={:+.5f}+-{:.5f} t={:+5.2f} | '
                     'B {:.5f} d={:+.5f}+-{:.5f} t={:+5.2f} | both {:.5f} d={:+.5f}+-{:.5f}'
                     .format(var,
                             s.loc['A', 'rps'], s.loc['A', 'd'], s.loc['A', 'ci95'], s.loc['A', 't'],
                             s.loc['B', 'rps'], s.loc['B', 'd'], s.loc['B', 'ci95'], s.loc['B', 't'],
                             s.loc['both', 'rps'], s.loc['both', 'd'], s.loc['both', 'ci95']))
    print('\n'.join(lines))
    return t


# =====================================================================
def main():
    df = pd.read_csv(os.path.join(ROOT, 'data', 'matches.csv'))
    t0 = time.time()
    all_rows = []

    # ---------------- baseline -------------------------------------
    base = wf(df, ExpDC, BASE)
    A = base[base.date <= SPLIT]
    B = base[base.date > SPLIT]
    print('=== BASELINE (reproduction check) ===')
    print('A n={:d} RPS={:.5f}   B n={:d} RPS={:.5f}   both n={:d} RPS={:.5f}   [{:.1f}s]'
          .format(len(A), A.rps.mean(), len(B), B.rps.mean(), len(base), base.rps.mean(),
                  time.time() - t0))
    print('draw rate: model {:.3f}  actual {:.3f}   (A {:.3f}/{:.3f}  B {:.3f}/{:.3f})'
          .format(base.pD.mean(), (base.out == 1).mean(),
                  A.pD.mean(), (A.out == 1).mean(), B.pD.mean(), (B.out == 1).mean()))
    print('rho fitted per fold: mean {:+.4f} min {:+.4f} max {:+.4f}; folds with rho<0: {:.0%}'
          .format(base.rho_mle.mean(), base.rho_mle.min(), base.rho_mle.max(),
                  (base.rho_mle < 0).mean()))

    # ---------------- (b) rho: tau-only vs FULL likelihood ----------
    print('\n=== (b) rho: is the tau-only profile the right likelihood? ===')
    rho_full_check(df)

    print('\n--- (b) rho rules (re-scored, no refit) ---')
    rows = []
    for lab, rule in [('rho = 0 (independent)', 'zero'),
                      ('rho <= 0 constrained MLE', 'neg'),
                      ('rho = -0.03 fixed', -0.03),
                      ('rho = -0.06 fixed', -0.06),
                      ('rho = -0.10 fixed', -0.10),
                      ('rho = -0.15 (post-hoc opt)', -0.15),
                      ('rho = -0.20 (post-hoc opt)', -0.20),
                      ('rho = +0.06 fixed', 0.06)]:
        rows += paired(rescore(base, rho_rule=rule), base, lab)
    show(rows)
    all_rows += rows
    rho_scan(base)

    # ---------------- (a) count distribution ------------------------
    print('\n=== (a) count distribution on the goals stage (mean-matched) ===')
    rows = []
    for phi in (1.05, 1.15, 1.30):
        rows += paired(rescore(base, 'nb', phi), base, 'NB  Var={:.2f}*lam'.format(phi))
    for nu in (0.80, 0.90, 1.10, 1.25, 1.50):
        rows += paired(rescore(base, 'cmp', nu), base, 'CMP nu={:.2f} ({})'
                       .format(nu, 'over' if nu < 1 else 'under'))
    show(rows)
    all_rows += rows
    print('\n  in-sample dispersion of GOALS around the model lambda, per fold '
          '(training matches only):')
    fold_dispersion(df, base)
    draw_diag(base)

    # ---------------- (c) home/away split strengths -----------------
    print('\n=== (c) separate home / away team strengths, L2 shrinkage kappa ===')
    rows = []
    for kap in (30.0, 10.0, 3.0, 1.0):
        v = wf(df, ExpDC, dict(BASE, split_kappa=kap))
        rows += paired(v, base, 'split atk/def kappa={:.0f}'.format(kap))
    show(rows)
    all_rows += rows

    # ---------------- (d) per-team home advantage -------------------
    print('\n=== (d) per-team home advantage, L2 shrinkage kappa ===')
    rows = []
    for kap in (30.0, 10.0, 3.0):
        v = wf(df, ExpDC, dict(BASE, ha_kappa=kap))
        rows += paired(v, base, 'per-team HA kappa={:.0f}'.format(kap))
    show(rows)
    all_rows += rows

    # ---------------- (e) finishing ---------------------------------
    print('\n=== (e) team finishing (goals - xG persistence) ===')
    rows = []
    for w_, k0, side in ((1.0, 6.0, 'att'), (0.5, 6.0, 'att'), (1.0, 15.0, 'att'),
                         (1.0, 6.0, 'def'), (1.0, 6.0, 'both'), (0.5, 6.0, 'both')):
        v = wf(df, ExpDC, dict(BASE, fin_w=w_, fin_k0=k0, fin_side=side))
        rows += paired(v, base, 'finishing w={:.1f} k0={:.0f} {}'.format(w_, k0, side))
    show(rows)
    all_rows += rows

    finishing_persistence(df)

    out = pd.DataFrame(all_rows)
    out.to_csv(os.path.join(ROOT, 'data', 'exp_structure.csv'), index=False)
    print('\nsaved data/exp_structure.csv   total {:.0f}s'.format(time.time() - t0))


def rho_full_check(df):
    """Fit on ALL data once and compare: (i) the tau-only profile used by model.py,
    (ii) the full weighted DC log-likelihood over the whole score matrix."""
    played = df[df.played & df.hg.notna()]
    m = ExpDC(**BASE).fit(played)
    hi = played.home.map(m.idx).values
    ai = played.away.map(m.idx).values
    lh, la = m._lam_arrays(hi, ai)
    hg = played.hg.values.astype(int)
    ag = played.ag.values.astype(int)
    w = np.exp(-np.log(2) / m.half_life *
               np.maximum((played.ts.max() - played.ts.values) / 86400.0, 0.0))
    full = []
    for rho in RHO_GRID:
        s = 0.0
        for i in range(len(played)):
            M = score_matrix(lh[i], la[i], rho)
            x, y = min(hg[i], MAXG), min(ag[i], MAXG)
            s += w[i] * np.log(max(M[x, y], 1e-300))
        full.append(-s)
    full = np.asarray(full)
    tau = m.rho_profile
    print('  tau-only  argmin rho = {:+.3f}'.format(RHO_GRID[int(np.argmin(tau))]))
    print('  FULL DC   argmin rho = {:+.3f}   (identical up to grid: the Poisson part '
          'of the DC likelihood does not depend on rho, and tau conserves total mass)'
          .format(RHO_GRID[int(np.argmin(full))]))
    lo = full.min()
    # curvature -> approximate SE from the full profile
    j = int(np.argmin(full))
    if 0 < j < len(full) - 1:
        h = (full[j + 1] - 2 * full[j] + full[j - 1]) / (RHO_GRID[1] - RHO_GRID[0]) ** 2
        print('  SE(rho) from profile curvature = {:.3f}  ->  rho is {:.1f} SE from 0'
              .format(1 / np.sqrt(max(h, 1e-9)), abs(RHO_GRID[j]) * np.sqrt(max(h, 1e-9))))
    d2 = full - lo
    inside = RHO_GRID[d2 <= 1.92]
    print('  95% profile-likelihood interval for rho: [{:+.3f}, {:+.3f}]'
          .format(inside.min(), inside.max()))
    print('  low-score cells, observed vs Poisson-expected (all {} matches):'.format(len(played)))
    for (x, y) in ((0, 0), (0, 1), (1, 0), (1, 1)):
        obs = float(np.mean((hg == x) & (ag == y)))
        exp_ = float(np.mean(poisson.pmf(x, lh) * poisson.pmf(y, la)))
        print('    {}-{}: obs {:.4f}  exp {:.4f}  ratio {:.2f}'.format(x, y, obs, exp_, obs / exp_))


def fold_dispersion(df, base):
    """Per-fold in-sample Pearson dispersion and CMP/NB MLE of goals around lambda."""
    played = df[df.played & df.hg.notna()].sort_values('ts').reset_index(drop=True)
    days = sorted(played.loc[played.date >= START, 'date'].unique())
    pear, nus, phis = [], [], []
    for day in days[::10]:
        train = played[played.date < day]
        if len(train) < 120:
            continue
        m = ExpDC(**BASE).fit(train, ref_ts=played[played.date == day].ts.min())
        hi = train.home.map(m.idx).values
        ai = train.away.map(m.idx).values
        lh, la = m._lam_arrays(hi, ai)
        g = np.concatenate([train.hg.values, train.ag.values]).astype(float)
        lam = np.concatenate([lh, la])
        pear.append(np.sum((g - lam) ** 2 / lam) / (len(g) - 2 * len(m.teams) - 2))
        best, bn = None, None
        for nu in np.arange(0.30, 2.05, 0.05):
            P = cmp_pmf(lam, nu)
            ll = np.sum(np.log(np.maximum(P[np.arange(len(g)), np.minimum(g, MAXG).astype(int)], 1e-300)))
            if best is None or ll > best:
                best, bn = ll, nu
        nus.append(bn)
        bestp, bp = None, None
        for phi in np.arange(1.0, 3.05, 0.05):
            P = nb_pmf(lam, phi)
            ll = np.sum(np.log(np.maximum(P[np.arange(len(g)), np.minimum(g, MAXG).astype(int)], 1e-300)))
            if bestp is None or ll > bestp:
                bestp, bp = ll, phi
        phis.append(bp)
    print('    Pearson dispersion  mean {:.3f}  range [{:.3f}, {:.3f}]'
          .format(np.mean(pear), np.min(pear), np.max(pear)))
    print('    CMP nu MLE          mean {:.3f}  range [{:.2f}, {:.2f}]   (1 = Poisson, '
          '>1 = UNDER-dispersed)'.format(np.mean(nus), np.min(nus), np.max(nus)))
    print('    NB  phi MLE         mean {:.3f}  range [{:.2f}, {:.2f}]   (1 = Poisson, '
          'boundary => no overdispersion)'.format(np.mean(phis), np.min(phis), np.max(phis)))


def rho_scan(base):
    """Fine profile of the 1X2 scores as a function of a FIXED rho, on both windows.

    NOTE: the minimum of this curve is a post-hoc optimum measured ON the reported
    windows.  It is a description of the loss surface, NOT a recommended value --
    the implementable rules are 'rho = 0' and the constrained MLE 'rho <= 0'.
    """
    print('\n=== (b) fine profile over a fixed rho (post-hoc description, not a choice) ===')
    print('  rho     RPS_A     d_A       RPS_B     d_B       LL_A    LL_B    pD_A   pD_B')
    bA = base[base.date <= SPLIT]
    bB = base[base.date > SPLIT]
    for r in (0.03, 0.0, -0.05, -0.10, -0.15, -0.20, -0.25, -0.30):
        v = rescore(base, rho_rule=r)
        A, B = v[v.date <= SPLIT], v[v.date > SPLIT]
        print('  {:+.2f}  {:.5f}  {:+.5f}  {:.5f}  {:+.5f}  {:.4f}  {:.4f}  {:.3f}  {:.3f}'
              .format(r, A.rps.mean(), (A.rps.values - bA.rps.values).mean(),
                      B.rps.mean(), (B.rps.values - bB.rps.values).mean(),
                      A.ll.mean(), B.ll.mean(), A.pD.mean(), B.pD.mean()))


def draw_diag(base):
    """Where does the draw deficit come from, and how much is it worth at most?

    Each predictive variant is scored for the DRAW probability it produces, and an
    ORACLE draw-shift is fitted ON the evaluation window itself -- that is not a
    usable model, it is a CEILING on everything the count law and rho can buy.
    """
    print('\n=== draw calibration: mean P(draw) by predictive variant ===')
    acts = {'A': (base[base.date <= SPLIT].out == 1).mean(),
            'B': (base[base.date > SPLIT].out == 1).mean()}
    print('  actual draw rate            A {:.3f}   B {:.3f}'.format(acts['A'], acts['B']))
    for lab, kw in [('baseline (rho=MLE, Poisson)', {}),
                    ('rho = 0', dict(rho_rule='zero')),
                    ('rho = -0.10', dict(rho_rule=-0.10)),
                    ('NB  Var=1.30*lam', dict(kind='nb', par=1.30)),
                    ('CMP nu=0.80 (over)', dict(kind='cmp', par=0.80)),
                    ('CMP nu=1.25 (under)', dict(kind='cmp', par=1.25)),
                    ('CMP nu=1.50 (under)', dict(kind='cmp', par=1.50))]:
        v = rescore(base, **kw)
        print('  {:<28s} A {:.3f}   B {:.3f}'.format(
            lab, v[v.date <= SPLIT].pD.mean(), v[v.date > SPLIT].pD.mean()))

    print('\n=== where the draw deficit actually sits (250 out-of-sample matches) ===')
    hg, ag = base.hg.values, base.ag.values
    lh, la, rh = base.lh.values, base.la.values, base.rho.values
    to = te = 0.0
    for k in range(5):
        o = float(np.mean((hg == k) & (ag == k)))
        e = float(np.mean([matrix_from(lh[i], la[i], rh[i])[k, k] for i in range(len(base))]))
        to += o; te += e
        star = '   <-- DC tau reaches this cell' if k <= 1 else '   <-- tau cannot touch this cell'
        print('  {}-{}  obs {:.4f}  exp {:.4f}  diff {:+.4f}{}'.format(k, k, o, e, o - e, star))
    print('  all draws  obs {:.4f}  exp {:.4f}  diff {:+.4f}'.format(to, te, to - te))

    print('\n=== ORACLE ceiling: best constant draw-shift, fitted on the window itself ===')
    for wn, part in (('A', base[base.date <= SPLIT]), ('B', base[base.date > SPLIT])):
        p = part[['pH', 'pD', 'pA']].values
        o = part.out.values
        best = (0.0, np.mean([rps(p[i], o[i]) for i in range(len(o))]))
        base_r = best[1]
        for k in np.arange(0.0, 0.121, 0.005):
            q = p.copy()
            q[:, 1] = p[:, 1] + k
            q[:, 0] = p[:, 0] * (1 - k / (p[:, 0] + p[:, 2]))
            q[:, 2] = p[:, 2] * (1 - k / (p[:, 0] + p[:, 2]))
            r = np.mean([rps(q[i], o[i]) for i in range(len(o))])
            if r < best[1]:
                best = (k, r)
        print('  window {}: best shift +{:.3f} on P(draw) -> RPS {:.5f} vs {:.5f} '
              '(gain {:+.5f}) -- IN-SAMPLE ORACLE, not achievable'
              .format(wn, best[0], best[1], base_r, best[1] - base_r))

    print('\n=== out-of-sample count-law fit on the 250 evaluation matches (diagnostic) ===')
    for wn, part in (('A', base[base.date <= SPLIT]), ('B', base[base.date > SPLIT])):
        lam = np.concatenate([part.lh.values, part.la.values])
        g = np.concatenate([part.hg.values, part.ag.values]).astype(float)
        pear = np.sum((g - lam) ** 2 / lam) / len(g)
        gi = np.minimum(g, MAXG).astype(int)
        nus = np.arange(0.30, 2.05, 0.05)
        lls = [np.sum(np.log(np.maximum(cmp_pmf(lam, nu)[np.arange(len(g)), gi], 1e-300)))
               for nu in nus]
        phis = np.arange(1.0, 3.05, 0.05)
        llp = [np.sum(np.log(np.maximum(nb_pmf(lam, ph)[np.arange(len(g)), gi], 1e-300)))
               for ph in phis]
        print('  window {}: Pearson dispersion {:.3f} | CMP nu MLE {:.2f} | NB phi MLE {:.2f}'
              .format(wn, pear, nus[int(np.argmax(lls))], phis[int(np.argmax(llp))]))


def finishing_persistence(df):
    """Split-half check: does team finishing (goals - xG) persist across time?"""
    played = df[df.played & df.hg.notna()].sort_values('ts').reset_index(drop=True)
    half = len(played) // 2
    out = []
    for name, cols in (('xg_all', ('h_xg_all', 'a_xg_all')), ('xg (base)', ('h_xg', 'a_xg'))):
        c0, c1 = cols
        if c0 not in played.columns:
            continue
        vals = {}
        for tag, part in (('h1', played.iloc[:half]), ('h2', played.iloc[half:])):
            teams = sorted(set(part.home) | set(part.away))
            G = {t: 0.0 for t in teams}
            X = {t: 0.0 for t in teams}
            for _, r in part.iterrows():
                if pd.isna(r[c0]) or pd.isna(r[c1]):
                    continue
                G[r.home] += r.hg; X[r.home] += r[c0]
                G[r.away] += r.ag; X[r.away] += r[c1]
            vals[tag] = {t: np.log((G[t] + 6) / (X[t] + 6)) for t in teams}
        common = sorted(set(vals['h1']) & set(vals['h2']))
        a = np.array([vals['h1'][t] for t in common])
        b = np.array([vals['h2'][t] for t in common])
        r = np.corrcoef(a - a.mean(), b - b.mean())[0, 1]
        out.append((name, len(common), r, a.std(), b.std()))
    print('\n  finishing split-half persistence (first half of matches vs second):')
    for name, n, r, s1, s2 in out:
        print('    {:<10s} n_teams={:d}  corr={:+.3f}  sd(log ratio) {:.3f} / {:.3f}'
              .format(name, n, r, s1, s2))
    print('    (a 95% CI on r with 14 teams is roughly +-0.55: this test cannot '
          'resolve anything smaller.)')


if __name__ == '__main__':
    main()
