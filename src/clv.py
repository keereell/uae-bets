# -*- coding: utf-8 -*-
"""
ЖУРНАЛ СТАВОК И CLV (closing line value).

Зачем. Чтобы отличить перевес 3% от нуля по прибыли, нужны тысячи ставок
(при среднем кэфе 3.5 — около 10 700). По CLV хватает десятков: стандартное
отклонение прибыли на ставку около 1.0, а CLV — около 0.1, то есть сигнал
проявляется примерно в 100 раз быстрее.

CLV = (цена, которую взял) / (справедливая цена по закрытию Pinnacle) − 1

Положительный CLV означает, что к закрытию рынок сдвинулся в твою сторону.
В литературе это лучший доступный индикатор наличия перевеса: Buchdahl
показал соотношение почти один к одному между CLV и фактической доходностью
(наклон 1.0025, R² 0.952 на 87 960 парах коэффициентов).

Использование:
    python src/clv.py log      # записать текущие рекомендации в журнал
    python src/clv.py update   # обновить цены закрытия по ещё не начавшимся матчам
    python src/clv.py report   # свести итоги
"""
import sys, os, csv, datetime as dt
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from markets import DEVIG
from pricing import price_bet
import pinnacle
from sharp import pinnacle_constraints, fit_from_constraints, NAME_MAP
from teams import to_en

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LEDGER = os.path.join(ROOT, 'data', 'bets_log.csv')
COLS = ['logged_at', 'match_ru', 'match_en', 'kickoff', 'market', 'line', 'outcome',
        'price_taken', 'model_p', 'edge_pp', 'ev_model',
        'pin_fair_at_log', 'pin_fair_closing', 'closing_updated_at',
        'stake_pct', 'result', 'pnl']


def _now():
    return dt.datetime.now().strftime('%Y-%m-%d %H:%M')


def _sharp_matrices():
    try:
        pin = pinnacle.parse()
    except Exception as e:
        print('Pinnacle недоступен:', e)
        return {}
    out = {}
    for g in pin.values():
        c = pinnacle_constraints(g)
        if len(c) >= 5:
            key = (NAME_MAP.get(g['home'], g['home']), NAME_MAP.get(g['away'], g['away']))
            out[key] = dict(fit=fit_from_constraints(c), start=g.get('start'))
    return out


def _load():
    if os.path.exists(LEDGER):
        return pd.read_csv(LEDGER, encoding='utf-8-sig')
    return pd.DataFrame(columns=COLS)


def _save(df):
    df.to_csv(LEDGER, index=False, encoding='utf-8-sig')


def pin_fair(sharp, h, a, market, line, outcome, h_ru='', a_ru=''):
    """Справедливый коэффициент по линии Pinnacle для конкретного исхода."""
    ent = sharp.get((h, a))
    if not ent:
        return None, None
    M = ent['fit']['M']
    r = price_bet(M, market, line, outcome, h_ru, a_ru)
    if r is None or r[0] <= 1e-6:
        return None, ent.get('start')
    w, pu, l = r
    return 1.0 + l / w, ent.get('start')


def cmd_log():
    p = os.path.join(ROOT, 'data', 'picks.csv')
    if not os.path.exists(p):
        print('нет data/picks.csv — сначала запусти src/value_report.py')
        return
    picks = pd.read_csv(p)
    sharp = _sharp_matrices()
    led = _load()
    added = 0
    for _, r in picks.iterrows():
        m_ru = str(r['матч'])
        parts = [x.strip() for x in m_ru.split('—')]
        h_ru, a_ru = (parts + ['', ''])[:2]
        h, a = to_en(h_ru), to_en(a_ru)
        line = '' if pd.isna(r.get('линия')) else str(r.get('линия'))
        key = (m_ru, str(r['рынок']), line, str(r['исход']))
        exists = ((led['match_ru'] == key[0]) & (led['market'] == key[1]) &
                  (led['line'].fillna('').astype(str) == key[2]) &
                  (led['outcome'] == key[3])).any() if len(led) else False
        if exists:
            continue
        fair, start = pin_fair(sharp, h, a, str(r['рынок']), line, str(r['исход']), h_ru, a_ru)
        led.loc[len(led)] = dict(
            logged_at=_now(), match_ru=m_ru, match_en=f'{h} — {a}', kickoff=start or '',
            market=r['рынок'], line=line, outcome=r['исход'],
            price_taken=r['кэф'], model_p=r.get('p'), edge_pp=r.get('edge_pp'),
            ev_model=r.get('ev'), pin_fair_at_log=fair, pin_fair_closing=np.nan,
            closing_updated_at='', stake_pct=r.get('Келли_итог_%', r.get('Келли_%')),
            result='', pnl=np.nan)
        added += 1
    _save(led)
    print(f'записано новых ставок: {added}, всего в журнале: {len(led)}')


def cmd_update():
    led = _load()
    if led.empty:
        print('журнал пуст')
        return
    sharp = _sharp_matrices()
    upd = 0
    for i, r in led.iterrows():
        parts = [x.strip() for x in str(r['match_ru']).split('—')]
        h_ru, a_ru = (parts + ['', ''])[:2]
        h, a = to_en(h_ru), to_en(a_ru)
        line = '' if pd.isna(r['line']) else str(r['line'])
        fair, _ = pin_fair(sharp, h, a, str(r['market']), line, str(r['outcome']), h_ru, a_ru)
        if fair is not None:
            led.at[i, 'pin_fair_closing'] = fair
            led.at[i, 'closing_updated_at'] = _now()
            upd += 1
    _save(led)
    print(f'обновлено цен закрытия: {upd}. Запускай перед самым стартом матчей — '
          f'API Pinnacle отдаёт только открытые рынки.')


def cmd_report():
    led = _load()
    if led.empty:
        print('журнал пуст')
        return
    d = led.copy()
    d['fair'] = d.pin_fair_closing.fillna(d.pin_fair_at_log)
    d = d.dropna(subset=['fair', 'price_taken'])
    if d.empty:
        print('нет ставок с эталонной ценой Pinnacle')
        return
    d['clv'] = d.price_taken / d.fair - 1.0
    n = len(d)
    mean = d.clv.mean()
    se = d.clv.std(ddof=1) / np.sqrt(n) if n > 1 else np.nan
    print(f'ставок с эталоном: {n}')
    print(f'средний CLV: {100*mean:+.2f}%' + (f'  ±{100*1.96*se:.2f}% (95%)' if n > 1 else ''))
    print(f'доля ставок с положительным CLV: {(d.clv > 0).mean():.1%}')
    print()
    print(d[['match_ru', 'market', 'outcome', 'price_taken', 'fair', 'clv']]
          .assign(clv=lambda x: (100 * x.clv).round(2)).to_string(index=False))
    print()
    if n >= 30:
        t = mean / se
        print(f't-статистика: {t:+.2f}. |t| > 2 -> перевес статистически заметен.')
    else:
        need = 30 - n
        print(f'Для первого содержательного вывода нужно ещё около {need} ставок. '
              f'По прибыли на том же уровне уверенности понадобилось бы несколько тысяч.')
    done = led.dropna(subset=['pnl'])
    if len(done):
        print(f'\nсыгранных ставок: {len(done)}, ROI {100*done.pnl.mean():+.2f}%')


if __name__ == '__main__':
    cmd = sys.argv[1] if len(sys.argv) > 1 else 'report'
    {'log': cmd_log, 'update': cmd_update, 'report': cmd_report}.get(cmd, cmd_report)()
