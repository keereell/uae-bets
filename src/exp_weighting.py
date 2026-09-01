# -*- coding: utf-8 -*-
"""
exp_weighting.py — is the time-weighting / opponent-adjustment scheme optimal?

Four independent questions, all evaluated walk-forward on BOTH windows:
    window A: 2025-02-01 .. 2025-12-31   (115 matches)
    window B: 2026-01-01 ..              (135 matches)
and always reported as a PAIRED difference vs the baseline
    half_life=180, alpha=0.30, w_goals=0.0, promoted_shift=0.25,
    xg_cols=('h_xg_open','a_xg_open'), dispersion=None

  (a) shape of the time decay      : step / matches-ago / two-component
  (b) explicit season-boundary discount on top of the exponential decay
  (c) pre-adjusting xG for opponent strength before fitting (two-pass)
  (d) blend of level-state xG with open-play xG

Implementation notes
--------------------
* Nothing in src/ is modified.  Arbitrary match weights are injected into the
  unmodified DixonColes by rewriting the `ts` column of the *training copy*:
  the model computes  w = exp(-ln2/half_life * (ref_ts-ts)/86400), so setting
      ts' = ref_ts + 86400 * half_life/ln2 * ln(w_desired)
  reproduces any desired weight vector exactly, with every downstream use of
  the weights (stage 1, stage 2, rho) consistent.  Verified against the
  baseline to 1e-12.
* All weight vectors are normalised to max = 1 (the most recent match), so the
  L2 penalty `alpha` keeps the same meaning across schemes.
* NO LOOK-AHEAD: every quantity used for match i is computed from matches with
  an earlier date only.  The season calendar (which season a fixture belongs
  to, and when each season starts) is known before kick-off and is treated as
  such.  The level-xG rescaling factor is an expanding-window estimate.

Run:  python src/exp_weighting.py            (all parts, ~3 min)
      python src/exp_weighting.py a b        (selected parts)
"""
import os
import sys
import time

import numpy as np
import pandas as pd
from scipy.optimize import minimize

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)

from model import DixonColes, detect_newcomers          # noqa: E402
from markets import wdl, DEVIG                          # noqa: E402
from backtest import rps                                # noqa: E402

BASE = dict(half_life=180.0, alpha=0.30, w_goals=0.0, promoted_shift=0.25,
            xg_cols=('h_xg_open', 'a_xg_open'), dispersion=None)
START = '2025-02-01'
SPLIT = '2026-01-01'
LN2 = np.log(2.0)


# ----------------------------------------------------------------- модель
class WDC(DixonColes):
    """DixonColes с произвольным вектором весов матчей."""

    _w = None

    def set_weights(self, w):
        w = np.asarray(w, float)
        self._w = np.clip(w / max(float(w.max()), 1e-12), 1e-8, 1.0)
        return self

    def fit(self, d, ref_ts=None, newcomers=()):
        if self._w is not None:
            if len(self._w) != len(d):
                raise ValueError('weights/train length mismatch')
            d = d.copy()
            d['ts'] = ref_ts + 86400.0 * self.half_life / LN2 * np.log(self._w)
        return super().fit(d, ref_ts=ref_ts, newcomers=newcomers)


def restage2(m, d, w):
    """Пересчитать mu, gamma, rho на реальных голах при новых atk/def."""
    hi = d.home.map(m.idx).values
    ai = d.away.map(m.idx).values
    base_h = m.atk[hi] - m.dfn[ai]
    base_a = m.atk[ai] - m.dfn[hi]
    gh = d.hg.values.astype(float)
    ga = d.ag.values.astype(float)

    def nll(th):
        mu, gam = th
        eh = mu + gam + base_h
        ea = mu + base_a
        lh = np.exp(np.clip(eh, -6, 3))
        la = np.exp(np.clip(ea, -6, 3))
        return float(np.sum(w * (lh - gh * eh)) + np.sum(w * (la - ga * ea)))

    r = minimize(nll, [m.mu, m.gamma], method='L-BFGS-B',
                 options=dict(maxiter=600))
    m.mu, m.gamma = float(r.x[0]), float(r.x[1])
    m.beta = np.zeros(0)
    m.rho = m._fit_rho(d, hi, ai, w, np.zeros((len(d), 0)))
    return m


# ------------------------------------------------------------ walk-forward
def walk(df, make_model, start_date=START, end_date=None, min_train=120):
    """Как backtest.walk_forward, но модель строит переданная фабрика.

    make_model(train, ref_ts, newcomers, season) -> подогнанная модель.
    """
    nc = detect_newcomers(df)
    played = df[df.played & df.hg.notna()].sort_values('ts').reset_index(drop=True)
    mask = played.date >= start_date
    if end_date:
        mask &= played.date <= end_date
    rows = []
    for day in sorted(played.loc[mask, 'date'].unique()):
        train = played[played.date < day]
        if len(train) < min_train:
            continue
        test = played[played.date == day]
        ref_ts = test.ts.min()
        new_here = set()
        for s in test.season.unique():
            new_here |= nc.get(s, set())
        season = int(test.season.iloc[0])
        try:
            m = make_model(train, ref_ts, new_here, season)
        except Exception as e:                                   # noqa: BLE001
            print('  fit fail', day, repr(e))
            continue
        for _, r in test.iterrows():
            if r.home not in m.idx or r.away not in m.idx:
                continue
            M, lh, la = m.matrix(r.home, r.away)
            p = wdl(M)
            out = 0 if r.hg > r.ag else (1 if r.hg == r.ag else 2)
            pm = np.array([p['H'], p['D'], p['A']])
            row = dict(date=r.date, home=r.home, away=r.away, out=out,
                       pH=pm[0], pD=pm[1], pA=pm[2],
                       ll=-np.log(max(pm[out], 1e-12)), rps=rps(pm, out))
            if not any(pd.isna([r.odds_H, r.odds_D, r.odds_A])):
                q = DEVIG['shin']([r.odds_H, r.odds_D, r.odds_A])
                row['rps_book'] = rps(q, out)
            rows.append(row)
    res = pd.DataFrame(rows)
    res['key'] = res.date + '|' + res.home + '|' + res.away
    return res


# ---------------------------------------------------------------- отчёт
def paired(res, base, label):
    """Строка отчёта: RPS по окнам A/B и парная разница с базой."""
    out = {'scheme': label}
    j = res.merge(base[['key', 'rps']].rename(columns={'rps': 'rps0'}), on='key')
    if len(j) != len(base):
        out['warn'] = 'matched %d/%d' % (len(j), len(base))
    for tag, sub in (('A', j[j.date < SPLIT]), ('B', j[j.date >= SPLIT])):
        d = (sub.rps - sub.rps0).values
        n = len(d)
        se = d.std(ddof=1) / np.sqrt(n) if n > 1 else np.nan
        out['rps_' + tag] = sub.rps.mean()
        out['d_' + tag] = d.mean()
        out['ci_' + tag] = 1.96 * se
        out['t_' + tag] = d.mean() / se if se else np.nan
        out['n_' + tag] = n
    d = (j.rps - j.rps0).values
    out['rps_all'] = j.rps.mean()
    out['d_all'] = d.mean()
    out['ci_all'] = 1.96 * d.std(ddof=1) / np.sqrt(len(d))
    return out


def show(rows, title):
    t = pd.DataFrame(rows)
    print('\n' + '=' * 108)
    print(title)
    print('=' * 108)
    print('%-42s %8s %9s %6s %8s %9s %6s' %
          ('scheme', 'RPS_A', 'd_A+-ci', 't_A', 'RPS_B', 'd_B+-ci', 't_B'))
    for _, r in t.iterrows():
        print('%-42s %8.5f %+7.5f+-%5.5f %5.2f %8.5f %+7.5f+-%5.5f %5.2f' %
              (r.scheme, r.rps_A, r.d_A, r.ci_A, r.t_A,
               r.rps_B, r.d_B, r.ci_B, r.t_B))
    return t


# ------------------------------------------------------- веса: схемы (a),(b)
def w_exp(dt_days, hl):
    return np.exp(-LN2 / hl * np.maximum(dt_days, 0.0))


def make_weight_factory(wfun, params=None):
    """wfun(train, ref_ts, season) -> веса; возвращает фабрику для walk()."""
    def factory(train, ref_ts, newcomers, season):
        m = WDC(**(params or BASE))
        m.set_weights(wfun(train, ref_ts, season))
        return m.fit(train, ref_ts=ref_ts, newcomers=newcomers)
    return factory


# ------------------------------------------------------------------ main
def main(parts):
    df = pd.read_csv(os.path.join(ROOT, 'data', 'matches.csv'))

    # календарь сезонов известен заранее (расписание), это не look-ahead
    sstart = df.groupby('season').ts.min().to_dict()

    def dt_days(train, ref_ts):
        return np.maximum((ref_ts - train.ts.values) / 86400.0, 0.0)

    # ---------------- базовая линия
    t0 = time.time()
    base = walk(df, make_weight_factory(
        lambda tr, ref, s: w_exp(dt_days(tr, ref), 180.0)))
    print('baseline  all %.5f  A %.5f  B %.5f  (n=%d, %.0fs)' % (
        base.rps.mean(),
        base[base.date < SPLIT].rps.mean(),
        base[base.date >= SPLIT].rps.mean(), len(base), time.time() - t0))
    bk = base.dropna(subset=['rps_book'])
    print('bet365    all %.5f  A %.5f  B %.5f  (n=%d)' % (
        bk.rps_book.mean(),
        bk[bk.date < SPLIT].rps_book.mean(),
        bk[bk.date >= SPLIT].rps_book.mean(), len(bk)))

    tables = {}

    # =============================================== (a) форма затухания
    if 'a' in parts:
        rows = []
        # (i) ступенька: текущий сезон = 1, всё прежнее = s
        for s in (0.20, 0.35, 0.50, 0.65, 0.80):
            def wf(tr, ref, cur, s=s):
                return np.where(tr.season.values == cur, 1.0, s)
            rows.append(paired(walk(df, make_weight_factory(wf)), base,
                               'a-i  step  prior-season=%.2f' % s))
            print('  ...', rows[-1]['scheme'])
        # (ii) затухание по числу МАТЧЕЙ назад, а не по дням
        for hm in (40, 60, 90, 130, 180):
            def wf(tr, ref, cur, hm=hm):
                k = np.arange(len(tr))[::-1].astype(float)  # tr отсортирован по ts
                return 0.5 ** (k / hm)
            rows.append(paired(walk(df, make_weight_factory(wf)), base,
                               'a-ii matches-ago hl=%d matches' % hm))
            print('  ...', rows[-1]['scheme'])
        # (iii) двухкомпонентное затухание (быстрое + медленное)
        for fast, slow, p in ((30, 400, 0.3), (45, 365, 0.4),
                              (60, 500, 0.5), (90, 720, 0.5)):
            def wf(tr, ref, cur, fast=fast, slow=slow, p=p):
                dd = dt_days(tr, ref)
                return p * w_exp(dd, fast) + (1 - p) * w_exp(dd, slow)
            rows.append(paired(walk(df, make_weight_factory(wf)), base,
                               'a-iii two-comp %d/%d p=%.1f' % (fast, slow, p)))
            print('  ...', rows[-1]['scheme'])
        tables['a'] = show(rows, '(a) TIME DECAY SHAPE  [baseline = exp, half-life 180 days]')

    # ============================================ (b) граница сезона
    if 'b' in parts:
        rows = []
        for disc in (0.15, 0.30, 0.50):
            def wf(tr, ref, cur, disc=disc):
                w = w_exp(dt_days(tr, ref), 180.0)
                old = tr.ts.values < sstart[cur]
                return w * np.where(old, 1.0 - disc, 1.0)
            rows.append(paired(walk(df, make_weight_factory(wf)), base,
                               'b    season discount %d%%' % round(disc * 100)))
            print('  ...', rows[-1]['scheme'])
        # для контекста: та же идея, но с более коротким/длинным периодом
        for hl, disc in ((300.0, 0.30), (120.0, 0.30)):
            def wf(tr, ref, cur, hl=hl, disc=disc):
                w = w_exp(dt_days(tr, ref), hl)
                old = tr.ts.values < sstart[cur]
                return w * np.where(old, 1.0 - disc, 1.0)
            P = dict(BASE, half_life=hl)
            rows.append(paired(walk(df, make_weight_factory(wf, P)), base,
                               'b    hl=%d + season discount %d%%' % (hl, disc * 100)))
            print('  ...', rows[-1]['scheme'])
        tables['b'] = show(rows, '(b) SEASON BOUNDARY  [0%% discount = baseline]')

    # =================================== (c) предварительная поправка на соперника
    if 'c' in parts:
        rows = []
        for k in (0.5, 1.0):
            for mode in ('replace', 'sum'):
                def factory(train, ref_ts, newcomers, season, k=k, mode=mode):
                    w = w_exp(dt_days(train, ref_ts), 180.0)
                    m1 = WDC(**BASE).set_weights(w).fit(
                        train, ref_ts=ref_ts, newcomers=newcomers)
                    dbar = float(np.mean(m1.dfn))
                    hi = train.home.map(m1.idx).values
                    ai = train.away.map(m1.idx).values
                    tr = train.copy()
                    tr['_ah'] = tr.h_xg_open.values * np.exp(k * (m1.dfn[ai] - dbar))
                    tr['_aa'] = tr.a_xg_open.values * np.exp(k * (m1.dfn[hi] - dbar))
                    P = dict(BASE, xg_cols=('_ah', '_aa'))
                    m2 = WDC(**P).set_weights(w).fit(
                        tr, ref_ts=ref_ts, newcomers=newcomers)
                    if mode == 'sum':
                        assert m2.teams == m1.teams
                        m2.dfn = m2.dfn + k * (m1.dfn - dbar)
                        d = tr[tr.played & tr.hg.notna()]
                        restage2(m2, d, w)
                    return m2
                rows.append(paired(walk(df, factory), base,
                                   'c    pre-adjust k=%.1f %s' % (k, mode)))
                print('  ...', rows[-1]['scheme'])
        tables['c'] = show(rows, '(c) OPPONENT PRE-ADJUSTMENT OF xG  [baseline = none]')

    # ================================================ (d) поправка на счёт
    if 'd' in parts:
        # масштаб level-xG оценивается расширяющимся окном (только прошлое)
        g = df[df.played & df.h_xg_open.notna()].sort_values('ts')
        so = (g.h_xg_open + g.a_xg_open).values
        sl = (g.h_xg_level + g.a_xg_level).values
        co, cl = np.cumsum(so) - so, np.cumsum(sl) - sl        # строго до матча
        scale = (co + 10 * 2.0) / (cl + 10 * 1.0)              # слабый априор 2.0
        sc = pd.Series(scale, index=g.index).reindex(df.index)
        d2 = df.copy()
        for wl in (0.15, 0.30, 0.50):
            d2['h_blend_%d' % (wl * 100)] = (
                wl * d2.h_xg_level * sc + (1 - wl) * d2.h_xg_open)
            d2['a_blend_%d' % (wl * 100)] = (
                wl * d2.a_xg_level * sc + (1 - wl) * d2.a_xg_open)
        rows = []
        for wl in (0.15, 0.30, 0.50):
            P = dict(BASE, xg_cols=('h_blend_%d' % (wl * 100),
                                    'a_blend_%d' % (wl * 100)))
            f = make_weight_factory(
                lambda tr, ref, s: w_exp(dt_days(tr, ref), 180.0), P)
            rows.append(paired(walk(d2, f), base,
                               'd    blend level w=%.2f' % wl))
            print('  ...', rows[-1]['scheme'])
        # контроль: чистый level-xG (перемасштабированный)
        d2['h_lv'] = d2.h_xg_level * sc
        d2['a_lv'] = d2.a_xg_level * sc
        P = dict(BASE, xg_cols=('h_lv', 'a_lv'))
        f = make_weight_factory(
            lambda tr, ref, s: w_exp(dt_days(tr, ref), 180.0), P)
        rows.append(paired(walk(d2, f), base, 'd    blend level w=1.00 (pure)'))
        print('  ...', rows[-1]['scheme'])
        tables['d'] = show(rows, '(d) GAME-STATE BLEND  w*level_rescaled+(1-w)*open  [w=0 = baseline]')

    # ============ (z) диагностика: насколько модель вообще чувствительна
    if 'z' in parts:
        # предпосылка вопроса (a-ii): сколько матчей укладывается в 180 дней
        pl = df[df.played & df.hg.notna()].sort_values('ts').reset_index(drop=True)
        cnt = [(pl.ts > t - 180 * 86400).sum() - (pl.ts > t).sum()
               for t in pl.ts.values[120:]]
        print('\nmatches inside the previous 180 days: min %d  med %d  max %d'
              % (min(cnt), int(np.median(cnt)), max(cnt)))
        rows = []
        for hl in (45.0, 90.0, 360.0, 900.0, 1e9):
            P = dict(BASE, half_life=hl)
            f = make_weight_factory(
                lambda tr, ref, s, hl=hl: w_exp(dt_days(tr, ref), hl), P)
            rows.append(paired(walk(df, f), base,
                               'z    exp half-life %.0f d' % hl))
            print('  ...', rows[-1]['scheme'])
        tables['z'] = show(rows, '(z) SENSITIVITY ENVELOPE of the exponential half-life')

    if tables:
        out = pd.concat(tables.values(), ignore_index=True)
        p = os.path.join(ROOT, 'data', 'exp_weighting.csv')
        out.to_csv(p, index=False, encoding='utf-8-sig')
        print('\nsaved ->', p)
    return tables


if __name__ == '__main__':
    args = [a for a in sys.argv[1:] if a in 'abcdz']
    main(args or ['a', 'b', 'c', 'd'])
