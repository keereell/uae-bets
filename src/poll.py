# -*- coding: utf-8 -*-
"""
НЕПРЕРЫВНЫЙ ОПРОС ЛИНИЙ.

Почему цикл внутри задачи, а не частый cron. Замерено: GitHub исполняет
около 10% запрошенных запусков расписания -- 26 фактических прогонов за
7 дней против 259 запрошенных, провалы между проверками до 5.5 часов.
Самоперезапуск через repository_dispatch невозможен: GITHUB_TOKEN намеренно
не может запускать другие воркфлоу, а заводить личный токен ради этого --
лишний секрет в обмен на удобство.

Поэтому одна задача живёт до 5.5 часов (предел GitHub -- 6) и опрашивает
изнутри. Даже при 10% исполнения ежечасного расписания это даёт 2-3 запуска
в сутки по 5.5 часа, то есть 11-16 часов покрытия вместо нынешних минут.
Репозиторий публичный -- минуты Actions не тарифицируются.

Почему вообще важна частота. Ярус, который мы играем, живёт в окне, где
острая линия уже уехала, а мягкая ещё нет. Kaunitz и др. (2017): ставки за
1-5 часов до начала дали +8.5% против +3.5% на закрывающих линиях. Окно
меряется часами, пропущенная проверка -- это пропущенная ставка.

    python src/poll.py                     # один проход, ничего не шлёт
    python src/poll.py --minutes 330       # цикл на 5.5 часа
    python src/poll.py --minutes 330 --send --commit
"""
import os
import sys
import time
import argparse
import subprocess
import datetime as dt

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
ROOT = os.path.dirname(HERE)

from books import fetch_all, BETTABLE                              # noqa: E402
from consensus import (build_consensus, find_value, pinnacle_fair,
                       save_snapshot, THETA)                        # noqa: E402

STATE_SENT = os.path.join(ROOT, 'data', 'sent_value.json')
INTERVAL = 180.0          # секунд между проверками
COMMIT_EVERY = 20         # снимков между коммитами (~1 час)

# Опрашивать только когда есть ради чего: линия открывается за несколько
# суток до матча, а до этого снимки одинаковы и жгут только место.
LOOK_AHEAD_H = 72.0


def upcoming_soon(quotes, hours=LOOK_AHEAD_H):
    """Есть ли среди снятого матч, начинающийся в ближайшие `hours`?"""
    now = time.time()
    ks = [q.kickoff for q in quotes if q.kickoff]
    if not ks:
        return True          # контора не отдала время -- не отключаемся вслепую
    return any(now < k <= now + hours * 3600 for k in ks)


def _load_sent():
    import json
    try:
        with open(STATE_SENT, encoding='utf-8') as f:
            return json.load(f)
    except Exception:
        return {}


def _save_sent(d):
    import json
    tmp = STATE_SENT + '.tmp'
    with open(tmp, 'w', encoding='utf-8') as f:
        json.dump(d, f, ensure_ascii=False, indent=1)
    os.replace(tmp, STATE_SENT)


def fmt(v):
    return (f"<b>{v['home']} — {v['away']}</b>\n"
            f"  {v['sel']} @ <b>{v['price']:.2f}</b> в {v['book']}\n"
            f"  справедливая {v['fair_price']:.2f} по эталону «{v['ref']}»"
            f" ({v['n_books']} контор)\n"
            f"  перевес <b>{100*v['ev']:+.2f}%</b>")


def notify(hits, send):
    """
    Шлём только НОВОЕ и только заметно подорожавшее. Повторять одну и ту же
    находку каждые три минуты -- быстрый способ приучить себя не читать бота.
    """
    if not hits:
        return 0
    sent = _load_sent()
    fresh = []
    for v in hits:
        k = f"{v['home']}|{v['away']}|{v['sel']}"
        prev = sent.get(k, {}).get('ev', -9)
        if v['ev'] > prev + 0.01:
            fresh.append(v)
            sent[k] = dict(ev=float(v['ev']), price=float(v['price']),
                           book=v['book'], at=dt.datetime.now(dt.timezone.utc)
                           .strftime('%Y-%m-%dT%H:%MZ'))
    if not fresh:
        return 0
    _save_sent(sent)
    if not send:
        print('  (--send не задан, не отправляю)')
        return len(fresh)
    try:
        from telegram_sender import send_message
    except Exception:
        sys.path.insert(0, ROOT)
        from telegram_sender import send_message
    head = (f'💰 <b>Найден перевес по цене</b> ({len(fresh)})\n'
            f'<i>максимальная цена против справедливой; модель не участвует</i>\n\n')
    send_message(head + '\n\n'.join(fmt(v) for v in fresh))
    return len(fresh)


def git_commit(msg):
    """Коммитим снимки пачками: держать блокировку записи весь цикл нельзя."""
    try:
        subprocess.run(['git', 'add', 'data/odds_snapshots.jsonl.gz',
                        'data/sent_value.json'], cwd=ROOT, check=False)
        r = subprocess.run(['git', 'diff', '--cached', '--quiet'], cwd=ROOT)
        if r.returncode == 0:
            return False
        subprocess.run(['git', 'commit', '-m', msg], cwd=ROOT, check=True)
        for _ in range(3):
            subprocess.run(['git', 'pull', '--rebase', '--autostash',
                            'origin', 'main'], cwd=ROOT, check=False)
            if subprocess.run(['git', 'push', 'origin', 'main'],
                              cwd=ROOT).returncode == 0:
                return True
            time.sleep(5)
    except Exception as e:
        print(f'  коммит не прошёл: {e}', file=sys.stderr)
    return False


def one_pass(send=False, verbose=True):
    quotes, errs = fetch_all(verbose=verbose)
    if not quotes:
        print('  котировок нет вообще', file=sys.stderr)
        return None, 0, 0
    cons = build_consensus(quotes)
    pin = pinnacle_fair()
    val = find_value(quotes, cons, pin)
    hits = [v for v in val if v['hit']]
    n_books = len({q.book for q in quotes if q.book in BETTABLE})
    save_snapshot(quotes, errs)
    n_sent = notify(hits, send)
    if verbose:
        best = f"{100*val[0]['ev']:+.2f}%" if val else '—'
        print(f'  котировок {len(quotes)}, контор для ставки {n_books}, '
              f'матчей {len(cons)}, находок {len(hits)} (лучшее {best}), '
              f'отправлено {n_sent}')
    return quotes, len(hits), n_sent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--minutes', type=float, default=0.0,
                    help='сколько крутиться; 0 = один проход')
    ap.add_argument('--interval', type=float, default=INTERVAL)
    ap.add_argument('--send', action='store_true')
    ap.add_argument('--commit', action='store_true')
    a = ap.parse_args()

    deadline = time.time() + a.minutes * 60
    i = 0
    total_hits = 0
    while True:
        i += 1
        print(f'[{dt.datetime.now(dt.timezone.utc):%H:%M:%S}Z] проход {i}')
        try:
            quotes, n_hits, _ = one_pass(send=a.send, verbose=(i == 1))
            total_hits += n_hits
        except Exception as e:
            print(f'  проход упал: {type(e).__name__}: {e}', file=sys.stderr)
            quotes = None

        if a.commit and i % COMMIT_EVERY == 0:
            git_commit(f'снимки линий {dt.datetime.now(dt.timezone.utc):%Y-%m-%dT%H:%MZ} [skip ci]')

        if a.minutes <= 0 or time.time() >= deadline:
            break
        if quotes and not upcoming_soon(quotes):
            print('  ближайшие 72 часа матчей нет, выхожу')
            break
        time.sleep(max(5.0, a.interval))

    if a.commit:
        git_commit(f'снимки линий {dt.datetime.now(dt.timezone.utc):%Y-%m-%dT%H:%MZ} [skip ci]')
    print(f'\nпроходов {i}, находок всего {total_hits}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
