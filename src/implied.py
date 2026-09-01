# -*- coding: utf-8 -*-
"""
Восстановление «консенсуса самой линии»: подгонка (lambda_дом, lambda_гост, rho)
под ВСЕ основные рынки БЕТСИТИ одновременно (1X2, все линии тоталов, все форы,
обе забьют, индивидуальные тоталы) после снятия маржи.

Зачем. Букмекер выставляет основную линию осмысленно, а производные рынки
(точный счёт, комбинации, «точное число голов») чаще считает механически.
Если какой-то исход выпадает из собственной же линии букмекера — это
несогласованность внутри одной конторы, и она не требует, чтобы наша модель
была точнее рынка. Это самый защищённый вид валуя.
"""
import re
import numpy as np
from scipy.optimize import minimize

from model import score_matrix
from markets import wdl, totals, team_total, btts, asian_handicap, asian_total, DEVIG

NUM = r'([+-]?\d+(?:[.,]\d+)?)'


def _f(s):
    return float(str(s).replace(',', '.'))


def collect_constraints(rows, main):
    """
    Собирает пары/тройки взаимодополняющих исходов основных рынков
    и снимает с них маржу. -> список (kind, args, target_prob, weight)
    """
    cons = []

    # ---- 1X2
    if all(k in main for k in ('1', 'X', '2')):
        q = DEVIG['shin']([main['1'], main['X'], main['2']])
        for nm, pr in zip(('H', 'D', 'A'), q):
            cons.append(('wdl', nm, float(pr), 3.0))

    # ---- тоталы: собираем пары Мен/Бол по каждой линии
    tot = {}
    for r in rows:
        if r['market'] not in ('Тотал', 'Азиатский тотал'):
            continue
        m = re.search(NUM, r['line'] or '')
        if not m:
            continue
        L = _f(m.group(1))
        side = 'over' if r['outcome'].startswith('Бол') else ('under' if r['outcome'].startswith('Мен') else None)
        if side:
            tot.setdefault(L, {})[side] = r['price']
    if main.get('tot_line') is not None:
        tot.setdefault(main['tot_line'], {}).update(over=main['over'], under=main['under'])
    for L, d in tot.items():
        if 'over' in d and 'under' in d:
            q = DEVIG['mult']([d['over'], d['under']])
            cons.append(('total_over', L, float(q[0]), 2.0))

    # ---- форы: пары Ф1(x) / Ф2(-x)
    hcp = {}
    for r in rows:
        if r['market'] not in ('Фора', 'Азиатская фора'):
            continue
        m = re.match(r'Ф([12])\s*\(' + NUM + r'\)', r['outcome'])
        if not m:
            continue
        side, L = m.group(1), _f(m.group(2))
        key = L if side == '1' else -L
        hcp.setdefault(key, {})[side] = r['price']
    if main.get('h1_line') is not None:
        hcp.setdefault(main['h1_line'], {}).update({'1': main['h1']})
        hcp.setdefault(-main['h2_line'], {}).update({'2': main['h2']})
    for L, d in hcp.items():
        if '1' in d and '2' in d:
            q = DEVIG['mult']([d['1'], d['2']])
            cons.append(('hcp_home', L, float(q[0]), 2.0))

    # ---- обе забьют
    bt = {r['outcome']: r['price'] for r in rows if r['market'] == 'Обе забьют'}
    if 'Да' in bt and 'Нет' in bt:
        q = DEVIG['mult']([bt['Да'], bt['Нет']])
        cons.append(('btts', None, float(q[0]), 1.5))

    # ---- индивидуальные тоталы
    it = {}
    for r in rows:
        if r['market'] != 'Индивидуальный тотал':
            continue
        m = re.match(r'ИТ([12])\s*\(' + NUM + r'\)', r['line'] or '')
        if not m:
            continue
        key = (m.group(1), _f(m.group(2)))
        side = 'over' if r['outcome'].startswith('Бол') else ('under' if r['outcome'].startswith('Мен') else None)
        if side:
            it.setdefault(key, {})[side] = r['price']
    for (who, L), d in it.items():
        if 'over' in d and 'under' in d:
            q = DEVIG['mult']([d['over'], d['under']])
            cons.append(('tt_over', (who, L), float(q[0]), 1.0))

    return cons


def _prob(kind, arg, M):
    if kind == 'wdl':
        return wdl(M)[arg]
    if kind == 'total_over':
        o, p, u = asian_total(M, arg, over=True)
        return o / max(o + u, 1e-9)
    if kind == 'hcp_home':
        w, p, l = asian_handicap(M, arg, 'home')
        return w / max(w + l, 1e-9)
    if kind == 'btts':
        return btts(M)[0]
    if kind == 'tt_over':
        who, L = arg
        o, u, p = team_total(M, 'home' if who == '1' else 'away', L)
        return o / max(o + u, 1e-9)
    return None


def fit_implied(rows, main, rho0=0.03):
    """Подбирает (lh, la, rho) под линию букмекера. -> (lh, la, rho, отчёт)"""
    cons = collect_constraints(rows, main)
    if len(cons) < 4:
        return None

    def loss(th):
        lh, la, rho = np.exp(th[0]), np.exp(th[1]), th[2]
        try:
            M = score_matrix(lh, la, rho)
        except Exception:
            return 1e9
        s = 0.0
        for kind, arg, target, w in cons:
            p = _prob(kind, arg, M)
            if p is None:
                continue
            s += w * (p - target) ** 2
        return s

    best, bv = None, np.inf
    for start in ((0.3, 0.2), (0.6, -0.2), (-0.2, 0.6), (0.0, 0.0)):
        r = minimize(loss, [start[0], start[1], rho0], method='Nelder-Mead',
                     options=dict(maxiter=4000, xatol=1e-7, fatol=1e-10))
        if r.fun < bv:
            bv, best = r.fun, r
    lh, la, rho = float(np.exp(best.x[0])), float(np.exp(best.x[1])), float(best.x[2])
    rho = float(np.clip(rho, -0.25, 0.20))
    M = score_matrix(lh, la, rho)
    report = []
    for kind, arg, target, w in cons:
        p = _prob(kind, arg, M)
        report.append(dict(kind=kind, arg=arg, market=target, implied=p, diff=p - target))
    rmse = float(np.sqrt(np.mean([(r['diff']) ** 2 for r in report])))
    return dict(lh=lh, la=la, rho=rho, M=M, rmse=rmse, n_cons=len(cons), report=report)
