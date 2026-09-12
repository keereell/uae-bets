# -*- coding: utf-8 -*-
"""
НЕПРЕРЫВНЫЙ ОПРОС ЛИНИЙ.

Почему цикл внутри задачи, а не частый cron. Замерено: GitHub исполняет
около 10% запрошенных запусков расписания -- 26 фактических прогонов за
7 дней против 259 запрошенных, провалы между проверками до 5.5 часов.
Самоперезапуск ВОЗМОЖЕН и без личного токена: по документации GitHub события
workflow_dispatch и repository_dispatch создают запуск даже от GITHUB_TOKEN
(нужно permissions: actions: write). Воркфлоу в конце задачи запускает свою
же копию; очередь uae-poller держит её, пока текущая не закончится.

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
                       save_snapshot, THETA, SNAPSHOT_DIR)          # noqa: E402

STATE_SENT = os.path.join(ROOT, 'data', 'sent_value.json')
# Журнал ОТПРАВЛЕННЫХ сигналов ценового пути. В bets_log.csv пишет только
# xG-путь (clv.py), и первый настоящий сигнал (Хор-Факкан П1 @2.82, 11 сентября
# 2026) нигде не остался -- CLV по нему было не посчитать. Закрывающая цена
# берётся не живым запросом, а из снимков: последний снимок до начала матча.
SIGNALS = os.path.join(ROOT, 'data', 'price_signals.csv')
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

    Порядок важен: СНАЧАЛА отправить, ПОТОМ записать как отправленное, и только
    если отправка удалась. Первая версия записывала до отправки (и даже при
    выключенном --send): одна ошибка Telegram -- и находка считалась
    доставленной, а повтор требовал роста перевеса ещё на процент. Ключ
    включает дату матча: без неё запись прошлогодней встречи той же пары
    глушила бы свежий перевес. Старые записи вычищаются.
    """
    if not hits:
        return 0
    sent = _load_sent()
    now = time.time()
    # Записи без времени матча (старый формат) считаем просроченными: иначе
    # они жили бы вечно и глушили свежие находки.
    sent = {k: v for k, v in sent.items() if v.get('ko', 0) >= now - 86400}
    fresh = []
    for v in hits:
        ko = v.get('kickoff') or 0
        day = dt.datetime.fromtimestamp(ko, dt.timezone.utc).strftime('%Y-%m-%d') if ko else '?'
        k = f"{v['home']}|{v['away']}|{v['sel']}|{day}"
        prev = sent.get(k, {}).get('ev', -9)
        if v['ev'] > prev + 0.01:
            fresh.append((k, v))
    if not fresh:
        return 0
    if not send:
        print(f'  (--send не задан: {len(fresh)} находок не отправляю и не запоминаю)')
        return len(fresh)
    try:
        from telegram_sender import send_message
    except Exception:
        sys.path.insert(0, ROOT)
        from telegram_sender import send_message
    head = (f'💰 <b>Найден перевес по цене</b> ({len(fresh)})\n'
            f'<i>максимальная цена против справедливой; модель не участвует</i>\n\n')
    ok = send_message(head + '\n\n'.join(fmt(v) for _, v in fresh))
    if not ok:
        print('  Telegram не принял сообщение -- находки НЕ помечены отправленными',
              file=sys.stderr)
        return 0
    stamp = dt.datetime.now(dt.timezone.utc).strftime('%Y-%m-%dT%H:%MZ')
    for k, v in fresh:
        sent[k] = dict(ev=float(v['ev']), price=float(v['price']), book=v['book'],
                       ko=float(v.get('kickoff') or now), at=stamp)
    _save_sent(sent)
    _log_signals(fresh, stamp)
    return len(fresh)


def _log_signals(fresh, stamp):
    import csv
    new = not os.path.exists(SIGNALS)
    with open(SIGNALS, 'a', newline='', encoding='utf-8-sig') as f:
        w = csv.writer(f)
        if new:
            w.writerow(['sent_at', 'home', 'away', 'sel', 'book', 'price', 'fair_price',
                        'p_fair', 'ev', 'ev_worst', 'ref', 'n_books', 'kickoff'])
        for _, v in fresh:
            ko = v.get('kickoff')
            w.writerow([stamp, v['home'], v['away'], v['sel'], v['book'],
                        f"{v['price']:.3f}", f"{v['fair_price']:.4f}", f"{v['p_fair']:.5f}",
                        f"{v['ev']:.5f}", f"{v['ev_worst']:.5f}", v['ref'], v['n_books'],
                        dt.datetime.fromtimestamp(ko, dt.timezone.utc).strftime('%Y-%m-%dT%H:%MZ') if ko else ''])

def git_commit(msg):
    """
    Коммитим снимки пачками: держать блокировку записи весь цикл нельзя.

    Файлы стейджатся ПО ОДНОМУ и только существующие: `git add a b` при
    отсутствующем b не стейджит ничего, и первый прогон 9 сентября 2026
    так потерял все снимки. Результат rebase проверяется: незавершённый
    rebase оставлял репозиторий в подвешенном состоянии, и все последующие
    коммиты прогона пропадали. Каждая неудача печатается.
    -> True если отправлено (или нечего было), False если нет.
    """
    files = ['data/snapshots', 'data/sent_value.json', 'data/price_signals.csv']
    try:
        staged = 0
        for f in files:
            if os.path.exists(os.path.join(ROOT, f)):
                r = subprocess.run(['git', 'add', '-A', f], cwd=ROOT,
                                   capture_output=True, text=True)
                if r.returncode != 0:
                    print(f'  git add {f}: {r.stderr.strip()}', file=sys.stderr)
                else:
                    staged += 1
        if not staged:
            print('  коммит: нечего стейджить', file=sys.stderr)
            return False
        if subprocess.run(['git', 'diff', '--cached', '--quiet'], cwd=ROOT).returncode == 0:
            return True                      # нечего коммитить -- это не ошибка
        r = subprocess.run(['git', 'commit', '-m', msg], cwd=ROOT,
                           capture_output=True, text=True)
        if r.returncode != 0:
            print(f'  git commit: {r.stderr.strip()}', file=sys.stderr)
            return False
        for i in range(3):
            r = subprocess.run(['git', 'pull', '--rebase', '--autostash', 'origin', 'main'],
                               cwd=ROOT, capture_output=True, text=True)
            if r.returncode != 0:
                print(f'  rebase не прошёл: {r.stderr.strip()[:200]}', file=sys.stderr)
                subprocess.run(['git', 'rebase', '--abort'], cwd=ROOT, capture_output=True)
                time.sleep(5)
                continue
            r = subprocess.run(['git', 'push', 'origin', 'main'], cwd=ROOT,
                               capture_output=True, text=True)
            if r.returncode == 0:
                print('  снимки закоммичены и отправлены')
                return True
            print(f'  push {i+1}/3 не прошёл: {r.stderr.strip()[:200]}', file=sys.stderr)
            time.sleep(5)
    except Exception as e:
        print(f'  коммит не прошёл: {type(e).__name__}: {e}', file=sys.stderr)
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
              f'матчей {len(cons)}, Pinnacle {len(pin)}, находок {len(hits)} '
              f'(лучшее {best}), отправлено {n_sent}')
    if not pin:
        print('  ВНИМАНИЕ: эталон Pinnacle пуст, находки судятся только по консенсусу',
              file=sys.stderr)
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
    commit_failed = False
    empty_streak = 0
    while True:
        i += 1
        t0 = time.time()
        print(f'[{dt.datetime.now(dt.timezone.utc):%H:%M:%S}Z] проход {i}')
        try:
            quotes, n_hits, _ = one_pass(send=a.send, verbose=(i == 1))
            total_hits += n_hits
        except Exception as e:
            print(f'  проход упал: {type(e).__name__}: {e}', file=sys.stderr)
            quotes = None
        # Пусто три прохода подряд -- значит, все предстоящие матчи уже
        # начались (их отсекает fetch_all), а следующих с линией ещё нет.
        # 11 сентября 2026 прогон в таком состоянии крутился 5.5 часов и
        # плодил копии. Один-два пустых прохода могут быть сетевым сбоем,
        # три подряд -- нет. Строка ниже -- сигнал воркфлоу не запускать копию.
        empty_streak = empty_streak + 1 if not quotes else 0
        if a.minutes > 0 and empty_streak >= 3:
            print('  ближайшие 72 часа матчей нет, выхожу')
            break

        if a.commit and i % COMMIT_EVERY == 0:
            if not git_commit(f'снимки линий {dt.datetime.now(dt.timezone.utc):%Y-%m-%dT%H:%MZ} [skip ci]'):
                commit_failed = True

        if a.minutes <= 0 or time.time() >= deadline:
            break
        if quotes and not upcoming_soon(quotes):
            print('  ближайшие 72 часа матчей нет, выхожу')
            break
        # Спим ОСТАТОК интервала, а не весь интервал: сам проход занимает
        # ~150 с, и прежний код давал реальный шаг ~330 с вместо обещанных 180.
        time.sleep(max(5.0, a.interval - (time.time() - t0)))

    if a.commit:
        if not git_commit(f'снимки линий {dt.datetime.now(dt.timezone.utc):%Y-%m-%dT%H:%MZ} [skip ci]'):
            commit_failed = True
    print(f'\nпроходов {i}, находок всего {total_hits}')
    # Потерянные снимки -- это провал прогона, даже если опрос шёл: иначе
    # зелёная задача скрывала бы молчаливую потерю данных.
    return 1 if commit_failed else 0


if __name__ == '__main__':
    sys.exit(main())
