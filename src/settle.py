# -*- coding: utf-8 -*-
"""
Расчёт сыгравших ставок и сводка по журналу.

    python src/settle.py            # рассчитать всё, что сыграло
    python src/settle.py digest     # то же + текстовая сводка для Telegram

Как считается исход. Вместо отдельной таблицы правил берётся вырожденное
распределение счёта: вся вероятность в клетке фактического счёта. Тогда
та же функция price_bet, которой считались вероятности, возвращает
(1, 0, 0) при выигрыше, (0, 1, 0) при возврате и (0, 0, 1) при проигрыше —
а на четвертных линиях честные (0.5, 0, 0.5). Один и тот же проверенный код
и для прогноза, и для расчёта, без шанса разойтись.
"""
import os, sys
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
ROOT = os.path.dirname(HERE)

from model import MAXG
from pricing import price_bet
from teams import to_en

LEDGER = os.path.join(ROOT, 'data', 'bets_log.csv')
MATCHES = os.path.join(ROOT, 'data', 'matches.csv')


def outcome_matrix(hg, ag):
    """Вырожденное распределение: вся масса в фактическом счёте."""
    M = np.zeros((MAXG + 1, MAXG + 1))
    M[min(int(hg), MAXG), min(int(ag), MAXG)] = 1.0
    return M


def settle_one(market, line, outcome, hg, ag, price, home_ru='', away_ru=''):
    """-> (исход, прибыль на 1 ед. ставки) либо (None, None) если рынок не поддержан."""
    M = outcome_matrix(hg, ag)
    r = price_bet(M, market, '' if pd.isna(line) else str(line), outcome, home_ru, away_ru)
    if r is None:
        return None, None
    w, p, l = r
    pnl = w * (price - 1.0) - l
    if w >= 0.999:
        res = 'выигрыш'
    elif l >= 0.999:
        res = 'проигрыш'
    elif p >= 0.999:
        res = 'возврат'
    elif w > l:
        res = 'полувыигрыш'      # четвертная линия: половина зашла
    elif l > w:
        res = 'полупроигрыш'
    else:
        res = 'возврат'
    return res, float(pnl)


def load():
    if not os.path.exists(LEDGER):
        return None, None
    led = pd.read_csv(LEDGER, encoding='utf-8-sig')
    m = pd.read_csv(MATCHES)
    m = m[m.played.fillna(False) & m.hg.notna()]
    return led, m


def find_match(m, match_ru, kickoff=None, logged_at=None):
    """
    Тот САМЫЙ матч, на который была ставка, а не любая встреча этих команд.
    Без привязки ко времени расчёт нашёл бы прошлогоднюю игру той же пары
    и закрыл бы ещё не сыгранную ставку чужим результатом.
    Допуск по времени +-2 дня; если времени начала нет, берём первую
    встречу этих команд ПОСЛЕ момента записи ставки в журнал.
    """
    parts = [x.strip() for x in str(match_ru).split('—')]
    if len(parts) < 2:
        return None
    h, a = to_en(parts[0]), to_en(parts[1])
    if not h or not a:
        return None
    sub = m[(m.home == h) & (m.away == a)].sort_values('ts')
    if sub.empty:
        return None

    ts = None
    if isinstance(kickoff, str) and kickoff.strip():
        try:
            ts = pd.to_datetime(kickoff, utc=True, errors='coerce').timestamp()
        except Exception:
            ts = None
    if ts is not None and not pd.isna(ts):
        near = sub[(sub.ts - ts).abs() <= 2 * 86400]
        return None if near.empty else near.iloc[0]

    if isinstance(logged_at, str) and logged_at.strip():
        t0 = pd.to_datetime(logged_at, errors='coerce')
        if not pd.isna(t0):
            after = sub[sub.ts >= t0.timestamp() - 86400]
            return None if after.empty else after.iloc[0]
    return None


def settle(verbose=True):
    led, m = load()
    if led is None or led.empty:
        print('журнал пуст')
        return None
    # колонки могли прочитаться как float64 из пустого CSV -- строки в них не лягут
    for c in ('result', 'score'):
        if c not in led.columns:
            led[c] = ''
        led[c] = led[c].astype('object').where(led[c].notna(), '')
    if 'pnl' not in led.columns:
        led['pnl'] = np.nan
    led['pnl'] = pd.to_numeric(led['pnl'], errors='coerce')

    done = 0
    for i, r in led.iterrows():
        if isinstance(r.get('result'), str) and r['result'].strip():
            continue
        g = find_match(m, r['match_ru'], r.get('kickoff'), r.get('logged_at'))
        if g is None:
            continue
        parts = [x.strip() for x in str(r['match_ru']).split('—')]
        res, pnl = settle_one(r['market'], r.get('line'), r['outcome'],
                              g.hg, g.ag, float(r['price_taken']),
                              parts[0] if parts else '', parts[1] if len(parts) > 1 else '')
        if res is None:
            continue
        led.at[i, 'result'] = res
        led.at[i, 'pnl'] = pnl
        led.at[i, 'score'] = f'{int(g.hg)}:{int(g.ag)}'
        done += 1
        if verbose:
            print(f"  {r['match_ru']} {int(g.hg)}:{int(g.ag)} | {r['outcome']} "
                  f"@{r['price_taken']} -> {res} ({pnl:+.2f})")
    if done:
        led.to_csv(LEDGER, index=False, encoding='utf-8-sig')
    print(f'рассчитано ставок: {done}')
    return led


def digest(led=None):
    """Текстовая сводка для Telegram."""
    if led is None:
        led, _ = load()
    if led is None or led.empty:
        return 'Журнал ставок пуст.'
    d = led.copy()
    d['fair'] = d.pin_fair_closing.fillna(d.pin_fair_at_log) if 'pin_fair_closing' in d else np.nan
    with_ref = d.dropna(subset=['fair'])
    settled = d.dropna(subset=['pnl'])

    L = ['📊 <b>Сводка по журналу ставок</b>', '']
    L.append(f'Всего записано: <b>{len(d)}</b>, сыграло: <b>{len(settled)}</b>')

    if len(with_ref):
        clv = (with_ref.price_taken / with_ref.fair - 1.0)
        se = clv.std(ddof=1) / np.sqrt(len(clv)) if len(clv) > 1 else np.nan
        closed = int(with_ref.pin_fair_closing.notna().sum()) if 'pin_fair_closing' in with_ref else 0
        L.append(f'CLV: <b>{100*clv.mean():+.2f}%</b>'
                 + (f' ± {100*1.96*se:.2f}%' if len(clv) > 1 else '')
                 + f'  (по закрытию: {closed} из {len(with_ref)})')
        L.append(f'Доля с положительным CLV: {100*(clv > 0).mean():.0f}%')

    if len(settled):
        roi = settled.pnl.mean()
        se = settled.pnl.std(ddof=1) / np.sqrt(len(settled)) if len(settled) > 1 else np.nan
        wins = int((settled.result == 'выигрыш').sum())
        L.append('')
        L.append(f'ROI: <b>{100*roi:+.2f}%</b>'
                 + (f' ± {100*1.96*se:.1f}%' if len(settled) > 1 else ''))
        L.append(f'Заходов: {wins} из {len(settled)}, '
                 f'прибыль {settled.pnl.sum():+.2f} ед.')
        if 'ярус' in settled.columns:
            for t, g in settled.groupby(settled['ярус'].astype(str)):
                L.append(f'  ярус {t}: {len(g)} ставок, ROI {100*g.pnl.mean():+.1f}%')

    n = len(with_ref)
    if n < 30:
        L.append('')
        L.append(f'<i>Для первого содержательного вывода нужно ещё около {30-n} ставок. '
                 f'По прибыли на том же уровне уверенности понадобилось бы несколько тысяч, '
                 f'поэтому смотрим на CLV.</i>')
    return '\n'.join(L)


if __name__ == '__main__':
    led = settle()
    if len(sys.argv) > 1 and sys.argv[1] == 'digest':
        print()
        print(digest(led))
