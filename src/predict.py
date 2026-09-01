# -*- coding: utf-8 -*-
"""
Итоговый конвейер:
  1) walk-forward валидация с лучшими гиперпараметрами
  2) пост-калибровка (сжатие переуверенности)
  3) финальная модель на всех данных -> прогноз будущих матчей
  4) сопоставление с коэффициентами БЕТСИТИ -> поиск валуя
"""
import sys, os, json
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model import DixonColes, detect_newcomers, score_matrix
from markets import (wdl, double_chance, totals, team_total, btts, asian_handicap,
                     DEVIG, margin, kelly)
from backtest import walk_forward, summarize, rps
from calibrate import fit_shrink, apply_shrink
from parse_betcity import load_all
from pricing import evaluate
from teams import to_en

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
pd.set_option('display.width', 250)
pd.set_option('display.max_columns', 60)
pd.set_option('display.max_colwidth', 40)

# ---- пороги валуя (обоснование см. в отчёте)
MIN_ODDS, MAX_ODDS = 1.35, 6.0
EDGE_MAIN = 0.04       # 1X2, тоталы, форы, ДШ, ОЗ
EDGE_DERIV = 0.09      # производные: точный счёт, комбо, точное число голов
MAIN_MARKETS = {'1X2', 'Тотал', 'Азиатский тотал', 'Фора', 'Азиатская фора',
                'Двойной исход', 'Обе забьют', 'Индивидуальный тотал', 'Голы'}


def best_params():
    """
    Значения взяты ИЗ ТЕОРИИ, а не подобраны по сетке. Причина — проверка переноса:
    108 конфигураций оценены на двух непересекающихся окнах (2025-02..2025-12 и
    2026-01..сейчас). Корреляция качества между окнами оказалась ОТРИЦАТЕЛЬНОЙ
    (Пирсон -0.255, Спирмен -0.243), а конфигурация, выигравшая на первом окне,
    заняла на втором 105-е место из 108 и проиграла медиане 0.00213 RPS.
    То есть на выборке в ~120 матчей подбор гиперпараметров ловит шум.

      half_life = 180 -- порядка половины сезона, как в исходной работе
                         Диксона и Коулза и последующих переоценках
      alpha = 0.30    -- умеренное сжатие силы команд к среднему по лиге
      w_goals = 0.0   -- сила команд по xG, а не по голам. ЭТОТ выбор
                         воспроизводится на обоих окнах и с большим запасом:
                         0.18810 против 0.19951 у модели на голах
      promoted_shift = 0.25 -- мягкий априор «новичок слабее среднего»
      xg_cols = xG с игры -- пенальти (12.5% всего xG лиги) и стандарты
                         выброшены как наименее повторяемые составляющие

    Честная метрика на данных, не участвовавших в выборе (2026-01 и позже,
    135 матчей): RPS 0.18047 против 0.17958 у закрывающей линии Bet365.
    """
    return dict(half_life=180.0, alpha=0.30, w_goals=0.0, promoted_shift=0.25,
                xg_cols=('h_xg_open', 'a_xg_open'))


def main():
    df = pd.read_csv(os.path.join(ROOT, 'data', 'matches.csv'))
    params = best_params()
    print('=== ГИПЕРПАРАМЕТРЫ (подобраны по walk-forward RPS) ===')
    print('  ', params)

    # ---------- 1. валидация + калибровка
    wf = walk_forward(df, params)
    print('\n=== 1. ВАЛИДАЦИЯ ВПЕРЁД ПО ВРЕМЕНИ (модель обучается только на прошлом) ===')
    s = summarize(wf)
    print(f"  матчей в тесте: {s['n']}")
    print(f"  модель:  log-loss {s['logloss']:.4f}  RPS {s['rps']:.4f}")
    if 'rps_book' in s:
        print(f"  Bet365:  log-loss {s['logloss_book']:.4f}  RPS {s['rps_book']:.4f}  (на {s['n_book']} матчах)")

    cal = fit_shrink(wf.lh, wf.la, wf.hg, wf.ag)
    print('\n  калибровка:', {k: round(v, 4) for k, v in cal.items()})
    ah, aa = apply_shrink(wf.lh.values, wf.la.values, cal['k_s'], cal['k_d'], cal['c'], cal['s_mean'])
    ll, rp = [], []
    for i in range(len(wf)):
        M = score_matrix(ah[i], aa[i], 0.0)
        p = wdl(M)
        pm = np.array([p['H'], p['D'], p['A']])
        o = int(wf.out.iloc[i])
        ll.append(-np.log(max(pm[o], 1e-12)))
        rp.append(rps(pm, o))
    print(f"  после калибровки: log-loss {np.mean(ll):.4f}  RPS {np.mean(rp):.4f}")

    wf.to_csv(os.path.join(ROOT, 'data', 'walkforward.csv'), index=False, encoding='utf-8-sig')

    # ---------- 2. финальная модель
    nc = detect_newcomers(df)
    upcoming = df[~df.played].sort_values('ts')
    ref_ts = float(upcoming.ts.min())
    newc = nc[max(nc)]
    m = DixonColes(**{k: (v if k != 'half_life' else v) for k, v in params.items()})
    m.fit(df, ref_ts=ref_ts, newcomers=newc)
    print('\n=== 2. ФИНАЛЬНАЯ МОДЕЛЬ ===')
    print(f"  масштаб xG {m.xg_scale:.3f} | mu {m.mu:.3f} | преимущество поля ×{np.exp(m.gamma):.3f}"
          f" | rho {m.rho:.3f} | эффективная выборка {m.eff_n:.0f} матчей")
    print('\n  РЕЙТИНГИ (выше = сильнее; atk — атака, def — оборона):')
    print(m.ratings().round(3).to_string(index=False))

    # ---------- 3. прогноз будущих матчей
    print('\n=== 3. ПРОГНОЗ БЛИЖАЙШИХ МАТЧЕЙ ===')
    rows = []
    fixtures = {}
    for _, r in upcoming.iterrows():
        if r.home not in m.idx or r.away not in m.idx:
            continue
        lh, la = m.lambdas(r.home, r.away)
        ah, aa = apply_shrink(np.array([lh]), np.array([la]),
                              cal['k_s'], cal['k_d'], cal['c'], cal['s_mean'])
        lh2, la2 = float(ah[0]), float(aa[0])
        M = score_matrix(lh2, la2, m.rho)
        p = wdl(M)
        o25 = totals(M, 2.5)
        bt = btts(M)
        fixtures[(r.home, r.away)] = dict(M=M, lh=lh2, la=la2, date=r.date, round=r['round'])
        rows.append(dict(дата=r.date, время=f'{int(r.kickoff_hour):02d}:{int(round((r.kickoff_hour%1)*60)):02d}',
                         матч=f'{r.home} — {r.away}',
                         xG_дом=round(lh2, 2), xG_гост=round(la2, 2),
                         P1=round(p['H'], 3), PX=round(p['D'], 3), P2=round(p['A'], 3),
                         кэф1=round(1/p['H'], 2), кэфX=round(1/p['D'], 2), кэф2=round(1/p['A'], 2),
                         ТБ25=round(o25[0], 3), ОЗ=round(bt[0], 3)))
    fc = pd.DataFrame(rows)
    print(fc.to_string(index=False))
    fc.to_csv(os.path.join(ROOT, 'data', 'forecast.csv'), index=False, encoding='utf-8-sig')

    # ---------- 4. поиск валуя против БЕТСИТИ
    print('\n=== 4. СОПОСТАВЛЕНИЕ С БЕТСИТИ ===')
    all_val = []
    for head, mrows in load_all():
        h_ru, a_ru = head.get('home'), head.get('away')
        h, a = to_en(h_ru), to_en(a_ru)
        key = (h, a)
        if key not in fixtures:
            print(f'  !! не нашёл матч в расписании: {h_ru} — {a_ru} -> {h} — {a}')
            continue
        fx = fixtures[key]
        M = fx['M']
        main = head.get('main', {})

        # основная линия как отдельные строки
        mr = list(mrows)
        if all(k in main for k in ('1', 'X', '2')):
            for nm in ('1', 'X', '2'):
                mr.append(dict(market='1X2', line='', outcome=nm, price=main[nm]))
        if main.get('tot_line') is not None:
            mr.append(dict(market='Тотал', line=str(main['tot_line']), outcome='Мен', price=main['under']))
            mr.append(dict(market='Тотал', line=str(main['tot_line']), outcome='Бол', price=main['over']))
        if main.get('h1_line') is not None:
            sign = '+' if main['h1_line'] >= 0 else ''
            mr.append(dict(market='Фора', line='', outcome=f"Ф1 ({sign}{main['h1_line']:g})", price=main['h1']))
            sign = '+' if main['h2_line'] >= 0 else ''
            mr.append(dict(market='Фора', line='', outcome=f"Ф2 ({sign}{main['h2_line']:g})", price=main['h2']))

        ev = evaluate(M, mr, h_ru, a_ru)
        for e in ev:
            e['матч'] = f'{h} — {a}'
            e['дата'] = fx['date']
        all_val.extend(ev)

        p = wdl(M)
        if all(k in main for k in ('1', 'X', '2')):
            o = [main['1'], main['X'], main['2']]
            q = DEVIG['shin'](o)
            print(f"\n  {h_ru} — {a_ru} ({fx['date']}), маржа БЕТСИТИ на 1X2: {100*margin(o):.2f}%")
            print(f"    рынок  (без маржи): 1={q[0]:.3f}  X={q[1]:.3f}  2={q[2]:.3f}")
            print(f"    модель            : 1={p['H']:.3f}  X={p['D']:.3f}  2={p['A']:.3f}")

    V = pd.DataFrame(all_val)
    if V.empty:
        print('\nНет сопоставленных исходов.')
        return
    V['тип'] = np.where(V.market.isin(MAIN_MARKETS), 'основной', 'производный')
    V['порог'] = np.where(V['тип'] == 'основной', EDGE_MAIN, EDGE_DERIV)
    V.to_csv(os.path.join(ROOT, 'data', 'all_bets.csv'), index=False, encoding='utf-8-sig')

    sel = V[(V.price >= MIN_ODDS) & (V.price <= MAX_ODDS) & (V.ev > V['порог'])].copy()
    sel['kelly_%банка'] = [100 * kelly(r.p_win / max(1 - r.p_push, 1e-9), r.price / 1.0, frac=0.25, cap=0.02)
                           for r in sel.itertuples()]
    sel = sel.sort_values('ev', ascending=False)
    print(f'\n=== 5. ВАЛУЙНЫЕ СТАВКИ (из {len(V)} оценённых исходов) ===')
    if sel.empty:
        print('  Ничего не проходит фильтры.')
    else:
        show = sel[['матч', 'market', 'line', 'outcome', 'price', 'fair', 'p_win', 'edge_pct',
                    'тип', 'kelly_%банка']].copy()
        show.columns = ['матч', 'рынок', 'линия', 'исход', 'кэф', 'справедливый',
                        'P(выигр)', 'перевес %', 'тип', 'Келли %']
        print(show.round(3).to_string(index=False))
    sel.to_csv(os.path.join(ROOT, 'data', 'value_bets.csv'), index=False, encoding='utf-8-sig')
    print(f"\nСохранено: data/forecast.csv, data/all_bets.csv, data/value_bets.csv, data/walkforward.csv")


if __name__ == '__main__':
    main()
