# -*- coding: utf-8 -*-
"""
Эталонная оценка вероятностей по линии Pinnacle и поиск валуя в БЕТСИТИ.

Логика:
  1. Берём всю линию Pinnacle (1X2 + все линии тоталов + все азиатские форы,
     включая четвертные) и снимаем маржу.
  2. Подгоняем под неё распределение счёта (lambda_дом, lambda_гост, rho).
     Получается полная «острая» картина матча.
  3. Прогоняем КАЖДЫЙ исход БЕТСИТИ через это распределение и считаем EV.

Почему так, а не «модель против букмекера»: walk-forward показал, что наша
модель уступает рынку. А вот расхождение мягкой конторы с Pinnacle — это
проверяемый и хорошо документированный источник прибыли.
"""
import sys, os
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model import score_matrix, DixonColes, detect_newcomers
from markets import (wdl, asian_handicap, asian_total, btts, team_total,
                     DEVIG, margin, kelly)
from implied import fit_implied, _prob
from parse_betcity import load_all
from pricing import price_bet
from calibrate import fit_shrink, apply_shrink
from backtest import walk_forward
from predict import best_params, MAIN_MARKETS
from teams import to_en
import pinnacle
import re
from scipy.optimize import minimize

NUM = r'([+-]?\d+(?:[.,]\d+)?)'

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
pd.set_option('display.width', 250)
pd.set_option('display.max_columns', 60)
pd.set_option('display.max_colwidth', 44)

DEVIG2 = 'power'   # для двусторонних рынков
DEVIG3 = 'shin'    # для 1X2


def pinnacle_constraints(g):
    cons = []
    ml = g.get('moneyline') or {}
    if all(k in ml for k in ('home', 'draw', 'away')):
        q = DEVIG[DEVIG3]([ml['home'], ml['draw'], ml['away']])
        for nm, pr in zip(('H', 'D', 'A'), q):
            cons.append(('wdl', nm, float(pr), 3.0))
    for L, v in (g.get('totals') or {}).items():
        if 'over' in v and 'under' in v:
            q = DEVIG[DEVIG2]([v['over'], v['under']])
            cons.append(('total_over', float(L), float(q[0]), 2.0))
    for L, v in (g.get('spreads') or {}).items():
        if 'home' in v and 'away' in v:
            q = DEVIG[DEVIG2]([v['home'], v['away']])
            cons.append(('hcp_home', float(L), float(q[0]), 2.0))
    return cons


def fit_from_constraints(cons, rho0=0.0):
    def loss(th):
        lh, la, rho = np.exp(th[0]), np.exp(th[1]), th[2]
        try:
            M = score_matrix(lh, la, rho)
        except Exception:
            return 1e9
        s = 0.0
        for kind, arg, target, w in cons:
            p = _prob(kind, arg, M)
            if p is not None:
                s += w * (p - target) ** 2
        return s

    best, bv = None, np.inf
    for st in ((0.3, 0.2), (0.7, -0.3), (-0.3, 0.7), (0.0, 0.0)):
        r = minimize(loss, [st[0], st[1], rho0], method='Nelder-Mead',
                     options=dict(maxiter=5000, xatol=1e-8, fatol=1e-11))
        if r.fun < bv:
            bv, best = r.fun, r
    lh, la = float(np.exp(best.x[0])), float(np.exp(best.x[1]))
    rho = float(np.clip(best.x[2], -0.25, 0.20))
    M = score_matrix(lh, la, rho)
    rep = [dict(kind=k, arg=a, target=t, fitted=_prob(k, a, M)) for k, a, t, _ in cons]
    rmse = float(np.sqrt(np.mean([(r['fitted'] - r['target']) ** 2 for r in rep if r['fitted'] is not None])))
    return dict(lh=lh, la=la, rho=rho, M=M, rmse=rmse, n=len(cons), report=rep)


# ---------------------------------------------------------------------------
#            ПРЯМАЯ ЦЕНА PINNACLE ВМЕСТО ПОДГОНКИ, ГДЕ ОНА ЕСТЬ
# ---------------------------------------------------------------------------
DEVIG_DIRECT = 'power'      # см. обоснование в docstring ниже


def direct_fair(g, market, line, outcome):
    """
    Справедливая цена по ЛИНИИ Pinnacle напрямую, без подгонки распределения.

    Подгонка нужна только для рынков, которых у Pinnacle нет. Там же, где он
    котирует сам (1X2, тотал на этой линии, фора на этой линии), подгонка лишь
    добавляет собственную ошибку формы: на матче Шабаб — Аль-Джазира она дала
    EV -3.2%, а прямой де-виг -3.7...-5.0% по всем пяти методам.

    Метод снятия маржи -- степенной: на двусторонних рынках он и
    мультипликативный различаются в третьем знаке, а на трёхстороннем 1X2
    даёт середину разброса пяти методов.

    -> (win, push, lose) как у price_bet, либо None если прямой цены нет.
    """
    mk = (market or '').strip()
    oc = (outcome or '').strip()

    if mk == '1X2':
        ml = g.get('moneyline') or {}
        if not all(k in ml for k in ('home', 'draw', 'away')):
            return None
        q = DEVIG[DEVIG_DIRECT]([ml['home'], ml['draw'], ml['away']])
        i = {'1': 0, 'X': 1, '2': 2}.get(oc)
        if i is None:
            return None
        return (float(q[i]), 0.0, float(1 - q[i]))

    if mk in ('Тотал', 'Азиатский тотал'):
        m = re.search(NUM, str(line or ''))
        if not m:
            return None
        L = float(m.group(1).replace(',', '.'))
        v = (g.get('totals') or {}).get(L)
        if not v or 'over' not in v or 'under' not in v:
            return None
        q = DEVIG[DEVIG_DIRECT]([v['over'], v['under']])
        # прямая котировка даёт вероятность УЖЕ без возврата: на целой линии
        # Pinnacle её просто не выставляет, поэтому push здесь всегда 0
        if oc.startswith('Бол'):
            return (float(q[0]), 0.0, float(q[1]))
        if oc.startswith('Мен'):
            return (float(q[1]), 0.0, float(q[0]))
        return None

    if mk in ('Фора', 'Азиатская фора'):
        m = re.match(r'Ф([12])\s*\(' + NUM + r'\)', oc)
        if not m:
            return None
        side, L = m.group(1), float(m.group(2).replace(',', '.'))
        # у Pinnacle линия форы записана со стороны ХОЗЯЕВ
        key = L if side == '1' else -L
        v = (g.get('spreads') or {}).get(key)
        if not v or 'home' not in v or 'away' not in v:
            return None
        q = DEVIG[DEVIG_DIRECT]([v['home'], v['away']])
        i = 0 if side == '1' else 1
        return (float(q[i]), 0.0, float(q[1 - i]))

    return None


def fair_ev(g, M_fit, market, line, outcome, price, h_ru='', a_ru=''):
    """
    Матожидание ставки против Pinnacle. Прямая цена, если Pinnacle торгует
    этот рынок сам; иначе — подгонка.

    -> (ev, источник, p_острая) либо (None, None, None), где p_острая --
    вероятность выигрыша ПРИ УСЛОВИИ «не возврат», выведенная из острой линии.
    Именно её, а не вероятность модели, следует подставлять в Келли: модель
    на этой лиге систематически расходится с рынком не в свою пользу
    (в каждом бакете расхождения факт ближе к рынку; Бриер 0.1902 против
    0.1846), и размер ставки по её вероятности завышен в разы.
    """
    r = direct_fair(g, market, line, outcome) if g else None
    src = 'прямая'
    if r is None:
        if M_fit is None:
            return None, None, None
        r = price_bet(M_fit, market, line, outcome, h_ru, a_ru)
        src = 'подгонка'
    if r is None or r[0] <= 1e-6:
        return None, None, None
    w, pu, l = r
    return w * price - (1 - pu), src, w / max(w + l, 1e-9)


NAME_MAP = {
    'Al Wasl': 'Al Wasl', 'Khor Fakkan Club': 'Khor Fakkan',
    'Shabab Al Ahli': 'Shabab Al Ahly', 'Al Jazira': 'Jazira Abu Dhabi',
    'Al Ittihad Kalba': 'Kalba', 'Al Ain': 'Al Ain',
    'Al-Nasr Dubai': 'Al Nasr Dubai', 'Ajman': 'Ajman Club',
    'Al Wahda': 'Al-Wahda', 'Sharjah': 'Sharjah SC', 'Bani Yas': 'Baniyas',
    'Al Dhafra': 'Al Dhafra', 'Hatta': 'Hatta Club', 'Dubai United': 'Dubai United',
}


def main():
    # ---------- наша модель (для третьего независимого мнения)
    df = pd.read_csv(os.path.join(ROOT, 'data', 'matches.csv'))
    params = best_params()
    wf = walk_forward(df, params)
    cal = fit_shrink(wf.lh, wf.la, wf.hg, wf.ag)
    cal['k_s'] = float(np.clip(cal['k_s'], 0.0, 1.5))
    cal['k_d'] = float(np.clip(cal['k_d'], 0.3, 1.5))
    nc = detect_newcomers(df)
    up = df[~df.played].sort_values('ts')
    mdl = DixonColes(**params).fit(df, ref_ts=float(up.ts.min()), newcomers=nc[max(nc)])

    # ---------- Pinnacle
    pin = pinnacle.parse()
    sharp = {}
    print('=' * 112)
    print('ЭТАЛОННАЯ ЛИНИЯ PINNACLE (подгонка распределения счёта)')
    for g in pin.values():
        cons = pinnacle_constraints(g)
        if len(cons) < 5:
            continue
        f = fit_from_constraints(cons)
        h = NAME_MAP.get(g['home'], g['home'])
        a = NAME_MAP.get(g['away'], g['away'])
        sharp[(h, a)] = f
        p = wdl(f['M'])
        ml = g['moneyline']
        print(f"\n  {g['home']} — {g['away']}")
        print(f"    Pinnacle 1X2: {ml['home']:.3f} / {ml['draw']:.3f} / {ml['away']:.3f}"
              f"  (маржа {100*(sum(1/v for v in ml.values())-1):.2f}%)")
        print(f"    подгонка: λ {f['lh']:.2f} / {f['la']:.2f}  rho {f['rho']:+.3f}"
              f"  |  честные вероятности 1={p['H']:.3f} X={p['D']:.3f} 2={p['A']:.3f}")
        print(f"    ограничений {f['n']}, RMSE {100*f['rmse']:.2f} п.п.")

    # ---------- сравнение с БЕТСИТИ
    rows = []
    print('\n' + '=' * 112)
    print('СРАВНЕНИЕ БЕТСИТИ С PINNACLE')
    for head, brows in load_all():
        h_ru, a_ru = head.get('home'), head.get('away')
        h, a = to_en(h_ru), to_en(a_ru)
        if (h, a) not in sharp:
            print(f'  !! нет линии Pinnacle для {h_ru} — {a_ru}')
            continue
        S = sharp[(h, a)]['M']
        main_line = head.get('main', {})

        mr = list(brows)
        if all(k in main_line for k in ('1', 'X', '2')):
            for nm in ('1', 'X', '2'):
                mr.append(dict(market='1X2', line='', outcome=nm, price=main_line[nm]))
        if main_line.get('tot_line') is not None:
            mr.append(dict(market='Тотал', line=f"{main_line['tot_line']:g}", outcome='Мен', price=main_line['under']))
            mr.append(dict(market='Тотал', line=f"{main_line['tot_line']:g}", outcome='Бол', price=main_line['over']))
        for kl, ko, tag in (('h1_line', 'h1', 'Ф1'), ('h2_line', 'h2', 'Ф2')):
            if main_line.get(kl) is not None:
                v = main_line[kl]
                mr.append(dict(market='Фора', line='',
                               outcome=f"{tag} ({'+' if v >= 0 else ''}{v:g})", price=main_line[ko]))

        # модель
        lh, la = mdl.lambdas(h, a)
        aa, bb = apply_shrink(np.array([lh]), np.array([la]), cal['k_s'], cal['k_d'], cal['c'], cal['s_mean'])
        Mm = score_matrix(float(aa[0]), float(bb[0]), mdl.rho)

        for r in mr:
            ps = price_bet(S, r['market'], r['line'], r['outcome'], h_ru, a_ru)
            if ps is None or ps[0] <= 1e-6:
                continue
            w, pu, l = ps
            ev = w * r['price'] - (1 - pu)
            pm = price_bet(Mm, r['market'], r['line'], r['outcome'], h_ru, a_ru)
            ev_m = (pm[0] * r['price'] - (1 - pm[1])) if pm and pm[0] > 1e-6 else np.nan
            rows.append(dict(матч=f'{h_ru} — {a_ru}', рынок=r['market'], линия=r['line'],
                             исход=r['outcome'], кэф=r['price'],
                             p_pinnacle=w, fair_pinnacle=1 + l / w, ev_pinnacle=ev,
                             ev_модели=ev_m,
                             тип='основной' if r['market'] in MAIN_MARKETS else 'производный'))

    V = pd.DataFrame(rows)
    V.to_csv(os.path.join(ROOT, 'data', 'vs_pinnacle.csv'), index=False, encoding='utf-8-sig')
    print(f'\nоценено исходов БЕТСИТИ через линию Pinnacle: {len(V)}')

    print('\nсредний EV БЕТСИТИ по типам рынков (относительно честной линии Pinnacle):')
    g = V.groupby('рынок').agg(исходов=('ev_pinnacle', 'size'),
                               средний_EV=('ev_pinnacle', 'mean'),
                               лучший_EV=('ev_pinnacle', 'max')).sort_values('средний_EV', ascending=False)
    print(g.round(4).to_string())

    print('\n' + '=' * 112)
    print('ВАЛУЙНЫЕ СТАВКИ: кэф БЕТСИТИ выше честного по Pinnacle')
    val = V[(V.ev_pinnacle > 0) & (V['кэф'].between(1.30, 12.0))].sort_values('ev_pinnacle', ascending=False)
    if val.empty:
        print('  Ни одного положительного исхода.')
    else:
        val = val.copy()
        val['Келли_%'] = [100 * kelly(r.p_pinnacle, r.кэф, frac=0.25, cap=0.02, p_push=0.0) for r in val.itertuples()]
        print(val[['матч', 'рынок', 'линия', 'исход', 'кэф', 'fair_pinnacle',
                   'ev_pinnacle', 'ev_модели', 'тип', 'Келли_%']].head(40).round(4).to_string(index=False))
    val.to_csv(os.path.join(ROOT, 'data', 'value_vs_pinnacle.csv'), index=False, encoding='utf-8-sig')


if __name__ == '__main__':
    main()
