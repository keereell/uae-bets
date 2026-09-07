# -*- coding: utf-8 -*-
"""
БЕТСИТИ в общем контракте books.Quote -- тонкая обёртка над src/betcity_api.py.

ПОЧЕМУ ОБЁРТКА, А НЕ ЕЩЁ ОДИН ЗАПРОС. Вся сетевая часть уже написана и годами
работает в betcity_api.py: один GET на весь сайт

    https://ad.betcity.ru/d/off/events?rev=6&add=dep_events&template=1&ver=88&csn=ooca9s

и разбор ветки reply/sports/1/chmps/11183/evts/*/main. Дублировать это здесь
означало бы завести второй источник правды по эндпоинту и по id чемпионата.
Поэтому модуль делает ровно одно: зовёт snapshot(save=False) и перекладывает
её ключи рынков в канонические sel.

ЧТО ОТДАЁТ ЭТОТ ПУТЬ. Блок main -- это только ОСНОВНЫЕ рынки: 1X2, двойной
исход, один тотал и одна фора, итого ~10 котировок на матч (проверено на туре
10-13.09.2026: 5 матчей, 50 котировок). Альтернативных линий тут нет и не
будет -- они лежат в полной росписи матча, за которой надо ходить отдельным
запросом на каждый матч. Для нашей задачи это не потеря: маржа БЕТСИТИ на
производных рынках 13-58%, в отбор они всё равно не попадают.

ФОРА ЗДЕСЬ АЗИАТСКАЯ (возврат при точном попадании), поэтому sel_ah, а не
sel_eh. Арифметика на живой линии 11.09.2026:
  * Хаур-Факкан -- Банияс, 1X2 = 2.45/3.80/2.44, фора 0 стоит 1.85 у ОБЕИХ
    команд. Европейская фора 0 -- это чистая победа, стоила бы как исход
    (2.45); «ничья -- возврат» даёт честные 2.00, с маржой ~1.85. Сходится
    второе.
  * Пара F1/F2 всегда двухисходная: 1/1.85 + 1/1.85 = 1.081, ровно обычная
    маржа конторы. У европейской форы был бы третий исход («ничья с форой»),
    и сумма двух сторон вышла бы заметно меньше единицы.
Если контора когда-нибудь заведёт в main европейскую фору отдельной группой --
её надо добавлять сюда явным белым списком, а не переключать sel_ah на sel_eh.

ЛИНИЯ ФОРЫ УЖЕ СО СВОИМ ЗНАКОМ. В main у F1 стоит lv=-1, у парного F2 lv=+1
(Аль-Джазира -- Аль-Наср Дубай, 10.09.2026). То есть каждая сторона записана
со стороны своей команды, сводить пары вручную не нужно -- знак берём из lv.

ЧТО СЛОМАЕТСЯ ПЕРВЫМ
1. id чемпионата. betcity_api.UAE_CHAMP = '11183'. При смене сезона БЕТСИТИ
   заводит турнир заново с новым id, и snapshot() начнёт молча возвращать
   пустой список (parse_champ на отсутствующий chmps отдаёт []). Найти новый:
       import betcity_api as b
       r = b.fetch()
       {k: v.get('name') for k, v in r['sports']['1']['chmps'].items()
        if 'ОАЭ' in (v.get('name') or '')}
   и поправить UAE_CHAMP в betcity_api.py (здесь id не дублируется намеренно).
   ВНИМАНИЕ: рядом живёт первый дивизион ОАЭ -- часть имён там совпадает с
   премьер-лигой, и перепутанный id означает сравнение цен на РАЗНЫЕ матчи.
2. Имена групп рынков. Ключ в snapshot собран как '<имя группы>|<исход>(линия)'.
   Мы разбираем ТОЛЬКО четыре группы по точному имени (плюс запасной разбор по
   номеру группы: 69 -- исход, 71 -- фора, 72 -- тотал). Это защита, а не
   ограничение: появится «Тотал 1-го тайма» -- он не пролезет в тотал матча.
   Симптом «котировок стало вдвое меньше» = контора переименовала группу,
   смотреть сырые ключи через `python src/betcity_api.py`.
3. Имена команд русские, канон -- английский, мост -- norm_team. Весь тур
   10-13.09.2026 (10 имён) распознался без остатка. Новое написание = матч
   молча выпадает; лечится добавлением алиаса в src/books/__init__.py,
   а НЕ угадыванием здесь.

    python src/books/betcity.py     # сводка по текущему туру
"""
import time

try:
    from books import Quote, norm_team, sel_total, sel_ah, sel_eh, UA
except ImportError:  # прямой запуск файла: python src/books/betcity.py
    import os, sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from books import Quote, norm_team, sel_total, sel_ah, sel_eh, UA

try:
    import betcity_api
except ImportError:  # betcity_api лежит в src/, на уровень выше пакета books
    import os, sys
    _SRC = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if _SRC not in sys.path:
        sys.path.insert(0, _SRC)
    import betcity_api

BOOK = 'betcity'

# Ретраи (3 попытки с растущей паузой) и таймаут 90 с живут в betcity_api.fetch.
# Второй слой здесь не нужен: он дал бы 9 заходов на один и тот же мёртвый хост.
# Сетевую ошибку не глушим -- она летит наверх, сборщик отличит «контора не
# котирует лигу» (пустой список) от «мы не достучались» (исключение).

# Заголовки/User-Agent тоже принадлежат betcity_api (там свой HDRS с Referer на
# страницу чемпионата). UA импортирован по контракту пакета, здесь не нужен.
_ = UA
# Европейских фор в main нет -- см. арифметику в докстроке.
_ = sel_eh

# Группы рынков. Ключ -- точное имя группы из snapshot; запасной ключ -- номер
# группы (parse_champ подставляет его, если у группы нет поля name).
_GROUPS = {
    'Фактический исход': '1X2', '69': '1X2',
    'Двойной исход': 'DC',
    'Фора': 'AH', '71': 'AH',
    'Тотал': 'TOT', '72': 'TOT',
}

_1X2 = {'P1': '1', 'X': 'X', 'P2': '2'}
_DC = {'1X': '1X', '12': '12', 'X2': 'X2'}


def _outcome(body):
    """'Kf_F1' -> 'F1'. Префикс Kf_ у конторы встречается не у всех групп."""
    body = body.split('(')[0]           # линия уже есть в mk['line'], в ключе она дубль
    return body[3:] if body.startswith('Kf_') else body


def _sel(key, mk):
    """Ключ рынка БЕТСИТИ -> канонический sel, либо None, если рынок чужой."""
    group, _sep, body = key.partition('|')
    kind = _GROUPS.get(group)
    if kind is None:
        return None
    out = _outcome(body)
    line = mk.get('line')

    if kind == '1X2':
        return _1X2.get(out)
    if kind == 'DC':
        return _DC.get(out)
    if line is None:
        return None                     # тотал/фора без линии -- битая запись
    if kind == 'TOT':
        # Tb = «больше», Tm = «меньше». Индивидуальные тоталы сюда не попадают:
        # у них своя группа, а её имени нет в _GROUPS.
        if out.startswith('Tb'):
            return sel_total(True, line)
        if out.startswith('Tm'):
            return sel_total(False, line)
        return None
    if kind == 'AH':
        if out.startswith('F1'):
            return sel_ah(1, line)
        if out.startswith('F2'):
            return sel_ah(2, line)
    return None


def _kickoff(start):
    """
    '2026-09-10 17:40' -> unix-секунды.

    betcity_api сделал эту строку через datetime.fromtimestamp, то есть в
    ЛОКАЛЬНОЙ зоне машины -- mktime возвращает её обратно ровно. Теряются
    только секунды (у конторы они всегда нулевые) и час перевода стрелок раз
    в год, если машина живёт в зоне с DST.
    """
    if not start:
        return None
    try:
        return time.mktime(time.strptime(start, '%Y-%m-%d %H:%M'))
    except (ValueError, OverflowError):
        return None


def _quotes(game):
    """Один матч snapshot -> список Quote (пустой, если команда не наша)."""
    raw_home, raw_away = game.get('home') or '', game.get('away') or ''
    home, away = norm_team(raw_home), norm_team(raw_away)
    if not home or not away:
        return []                       # молодёжка/кубок/чужая лига -- молча мимо

    best, when = {}, _kickoff(game.get('start'))
    for key, mk in (game.get('markets') or {}).items():
        try:
            price = float(mk['kf'])
        except (KeyError, TypeError, ValueError):
            continue
        if price <= 1.0:                # снятый рынок или заглушка
            continue
        sel = _sel(key, mk)
        if sel is None:
            continue
        # Один исход из двух ключей (напр. фора 0 как Kf_F1(+0) и F1(0)) --
        # берём лучшую цену, ставить всё равно будем по максимуму.
        if price > best.get(sel, (0.0, 0))[0]:
            best[sel] = (price, mk.get('md') or 0)

    return [Quote(book=BOOK, home=home, away=away, sel=sel, price=price,
                  kickoff=when, raw_home=raw_home, raw_away=raw_away,
                  # md -- момент последнего изменения цены. Ради него всё и
                  # затевалось: «острая линия уехала, а эта стоит» видно только тут.
                  extra={'md': md, 'event_id': game.get('id')})
            for sel, (price, md) in best.items()]


def fetch():
    """-> list[Quote]. Пустой список = контора сейчас не котирует лигу."""
    out = []
    for game in betcity_api.snapshot(save=False):   # save=False: адаптер не пишет файлы
        out.extend(_quotes(game))
    return out


if __name__ == '__main__':
    import datetime as dt

    # Снимок берём ОДИН раз и на нём же считаем сводку: лишний заход к конторе
    # ради той же самой линии -- бессмысленный трафик.
    raw_games = betcity_api.snapshot(save=False)
    qs = [q for g in raw_games for q in _quotes(g)]
    games = {}
    for q in qs:
        games.setdefault((q.home, q.away), []).append(q)

    print(f'{BOOK}: матчей {len(games)}, котировок {len(qs)}')
    for (home, away), lst in sorted(games.items(),
                                    key=lambda kv: kv[1][0].kickoff or 0):
        q0 = lst[0]
        when = (dt.datetime.fromtimestamp(q0.kickoff).strftime('%Y-%m-%d %H:%M')
                if q0.kickoff else '?')
        age = [q.extra.get('md') or 0 for q in lst]
        age_h = (time.time() - max(age)) / 3600 if max(age) else None
        print(f'\n{home} — {away}   {when}   [{q0.raw_home} / {q0.raw_away}]'
              + (f'   цены не менялись {age_h:.1f} ч' if age_h and age_h > 0 else ''))
        d = {q.sel: q.price for q in lst}
        fam = {'1X2': [s for s in ('1', 'X', '2') if s in d],
               'ДШ': [s for s in ('1X', '12', 'X2') if s in d],
               'тоталы': sorted((s for s in d if s[0] in 'OU'),
                                key=lambda s: (float(s[1:]), s)),
               'форы': sorted((s for s in d if s.startswith('AH')),
                              key=lambda s: (s[2], float(s[3:])))}
        for name, sels in fam.items():
            if sels:
                print(f'  {name:<7} ({len(sels):>2}) ' +
                      '  '.join(f'{s}={d[s]:g}' for s in sels))

    # Имена, которые norm_team не узнал -- дыры в словаре алиасов, а не ошибка.
    unmapped = sorted({n for g in raw_games for n in (g.get('home'), g.get('away'))
                       if n and not norm_team(n)})
    print(f'\nматчей в линии конторы: {len(raw_games)}, из них наших: {len(games)}')
    print('не распознано norm_team: ' + (', '.join(unmapped) if unmapped else 'нет'))
