# -*- coding: utf-8 -*-
"""
Итоговый поиск валуя. Каждый исход БЕТСИТИ оценивается ДВУМЯ независимыми способами:

  1) p_модели   — наша xG-модель Диксона-Коулза (внешняя оценка)
  2) p_линии    — распределение счёта, подогнанное под ОСНОВНЫЕ рынки самого
                  БЕТСИТИ после снятия маржи (внутренняя оценка)

Валуй по (2) означает, что букмекер противоречит сам себе — это самый надёжный
сигнал, он не требует, чтобы модель была точнее рынка.
Валуй по (1) требует, чтобы модель была права против рынка, — а walk-forward
показал, что это в среднем не так. Поэтому (1) используется только как фильтр
согласия, а не как самостоятельное основание.
"""
import sys, os
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model import DixonColes, detect_newcomers, score_matrix
from markets import wdl, totals, btts, DEVIG, margin, kelly
from backtest import walk_forward
from calibrate import fit_shrink, apply_shrink
from parse_betcity import load_all
from pricing import price_bet
from implied import fit_implied
from predict import best_params, MAIN_MARKETS
from teams import to_en

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
pd.set_option('display.width', 250)
pd.set_option('display.max_columns', 60)
pd.set_option('display.max_colwidth', 46)

MIN_ODDS, MAX_ODDS = 1.35, 7.0


def build_model(df):
    params = best_params()
    wf = walk_forward(df, params)
    cal = fit_shrink(wf.lh, wf.la, wf.hg, wf.ag)
    cal['k_s'] = float(np.clip(cal['k_s'], 0.0, 1.5))
    cal['k_d'] = float(np.clip(cal['k_d'], 0.3, 1.5))
    nc = detect_newcomers(df)
    upcoming = df[~df.played].sort_values('ts')
    m = DixonColes(**params).fit(df, ref_ts=float(upcoming.ts.min()), newcomers=nc[max(nc)])
    return m, cal, upcoming, wf


def main():
    df = pd.read_csv(os.path.join(ROOT, 'data', 'matches.csv'))
    m, cal, upcoming, wf = build_model(df)

    fixtures = {}
    for _, r in upcoming.iterrows():
        if r.home not in m.idx or r.away not in m.idx:
            continue
        lh, la = m.lambdas(r.home, r.away)
        a, b = apply_shrink(np.array([lh]), np.array([la]),
                            cal['k_s'], cal['k_d'], cal['c'], cal['s_mean'])
        fixtures[(r.home, r.away)] = dict(M=score_matrix(float(a[0]), float(b[0]), m.rho),
                                          lh=float(a[0]), la=float(b[0]), date=r.date)

    all_rows = []
    for head, rows in load_all():
        h_ru, a_ru = head.get('home'), head.get('away')
        h, a = to_en(h_ru), to_en(a_ru)
        if (h, a) not in fixtures:
            print(f'!! матч не найден в расписании: {h_ru} — {a_ru}')
            continue
        fx = fixtures[(h, a)]
        main_line = head.get('main', {})

        # добавляем основную линию как обычные исходы
        mr = list(rows)
        if all(k in main_line for k in ('1', 'X', '2')):
            for nm in ('1', 'X', '2'):
                mr.append(dict(market='1X2', line='', outcome=nm, price=main_line[nm]))
        if main_line.get('tot_line') is not None:
            mr.append(dict(market='Тотал', line=f"{main_line['tot_line']:g}", outcome='Мен', price=main_line['under']))
            mr.append(dict(market='Тотал', line=f"{main_line['tot_line']:g}", outcome='Бол', price=main_line['over']))
        for k_l, k_o, tag in (('h1_line', 'h1', 'Ф1'), ('h2_line', 'h2', 'Ф2')):
            if main_line.get(k_l) is not None:
                v = main_line[k_l]
                mr.append(dict(market='Фора', line='',
                               outcome=f"{tag} ({'+' if v >= 0 else ''}{v:g})", price=main_line[k_o]))

        imp = fit_implied(rows, main_line)
        Mi = imp['M'] if imp else None
        pm = wdl(fx['M'])
        pi = wdl(Mi) if Mi is not None else None
        print('=' * 110)
        print(f"{h_ru} — {a_ru}  ({fx['date']} {head.get('time')})")
        if all(k in main_line for k in ('1', 'X', '2')):
            o = [main_line['1'], main_line['X'], main_line['2']]
            print(f"  маржа БЕТСИТИ 1X2: {100*margin(o):.2f}%   кэфы {o}")
        print(f"  xG-модель      : λ {fx['lh']:.2f} / {fx['la']:.2f}  ->  "
              f"1={pm['H']:.3f} X={pm['D']:.3f} 2={pm['A']:.3f}")
        if imp:
            print(f"  линия БЕТСИТИ  : λ {imp['lh']:.2f} / {imp['la']:.2f}  ->  "
                  f"1={pi['H']:.3f} X={pi['D']:.3f} 2={pi['A']:.3f}"
                  f"   (подгонка по {imp['n_cons']} рынкам, RMSE {100*imp['rmse']:.2f} п.п.)")

        for r in mr:
            pmod = price_bet(fx['M'], r['market'], r['line'], r['outcome'], h_ru, a_ru)
            pimp = price_bet(Mi, r['market'], r['line'], r['outcome'], h_ru, a_ru) if Mi is not None else None
            if pmod is None and pimp is None:
                continue
            row = dict(матч=f'{h_ru} — {a_ru}', дата=fx['date'], рынок=r['market'],
                       линия=r['line'], исход=r['outcome'], кэф=r['price'])
            if pmod and pmod[0] > 1e-6:
                w, p_, l_ = pmod
                row['p_модели'] = w
                row['fair_модели'] = 1 + l_ / w
                row['ev_модели'] = w * r['price'] - (1 - p_)
            if pimp and pimp[0] > 1e-6:
                w, p_, l_ = pimp
                row['p_линии'] = w
                row['fair_линии'] = 1 + l_ / w
                row['ev_линии'] = w * r['price'] - (1 - p_)
            all_rows.append(row)

    V = pd.DataFrame(all_rows)
    V['тип'] = np.where(V['рынок'].isin(MAIN_MARKETS), 'основной', 'производный')
    V.to_csv(os.path.join(ROOT, 'data', 'all_bets.csv'), index=False, encoding='utf-8-sig')

    # ---------- сколько маржи букмекер закладывает в каждый тип рынка
    print('\n' + '=' * 110)
    print('СРЕДНЯЯ МАРЖА БЕТСИТИ ПО ТИПАМ РЫНКОВ (по её же собственной линии)')
    g = V.dropna(subset=['ev_линии']).groupby('рынок').agg(
        исходов=('ev_линии', 'size'), средний_ev=('ev_линии', 'mean'),
        лучший_ev=('ev_линии', 'max')).sort_values('средний_ev', ascending=False)
    print((g * 1).round(4).to_string())

    # ---------- кандидаты
    print('\n' + '=' * 110)
    print('КАНДИДАТЫ: исход дороже, чем следует из СОБСТВЕННОЙ линии БЕТСИТИ')
    c = V[(V['кэф'].between(MIN_ODDS, MAX_ODDS)) & (V['ev_линии'] > 0)].copy()
    c = c.sort_values('ev_линии', ascending=False)
    if c.empty:
        print('  Нет ни одного исхода с положительным EV относительно собственной линии букмекера.')
    else:
        show = c[['матч', 'рынок', 'линия', 'исход', 'кэф', 'fair_линии', 'ev_линии',
                  'fair_модели', 'ev_модели', 'тип']].head(30)
        print(show.round(4).to_string(index=False))

    # ---------- согласие двух оценок
    print('\n' + '=' * 110)
    print('СОГЛАСИЕ ДВУХ НЕЗАВИСИМЫХ ОЦЕНОК (и линия, и модель дают плюс)')
    both = V[(V['кэф'].between(MIN_ODDS, MAX_ODDS)) &
             (V['ev_линии'] > 0) & (V['ev_модели'] > 0.02)].copy()
    both['ev_мин'] = both[['ev_линии', 'ev_модели']].min(axis=1)
    both = both.sort_values('ev_мин', ascending=False)
    if both.empty:
        print('  Пусто.')
    else:
        both['Келли_%'] = [100 * kelly(r.p_линии, r.кэф, frac=0.25, cap=0.015)
                           for r in both.itertuples()]
        print(both[['матч', 'рынок', 'линия', 'исход', 'кэф', 'fair_линии', 'ev_линии',
                    'ev_модели', 'Келли_%']].round(4).to_string(index=False))
    both.to_csv(os.path.join(ROOT, 'data', 'value_bets.csv'), index=False, encoding='utf-8-sig')
    print('\nСохранено: data/all_bets.csv, data/value_bets.csv')


if __name__ == '__main__':
    main()
