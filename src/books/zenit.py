# -*- coding: utf-8 -*-
"""
Линия Zenit (zenit.win) по чемпионату ОАЭ.

Сайт -- react-приложение, которое ходит в свои же same-origin ajax-эндпоинты
и получает готовый JSON. Никакого html-парсинга не нужно, берём те же URL:

    /ajax/line/printer/react   -- список матчей лиги + главная линия
    /ajax/line/ross/data/      -- полная роспись по матчам (лестницы фор/тоталов)

ЗАГОЛОВОК imprintHash ОБЯЗАТЕЛЕН. Без него сервер отвечает
400 {"errorCode":400,"msg":"Не передан imprintHash в заголовках"}.
Значение при этом НЕ проверяется -- подходит любая ascii-строка (проверено:
'uae-line-reader' работает). Это защита от чужих скриптов «на отвяжись»,
но если её однажды сделают настоящей (подпись от фронта), адаптер начнёт
получать 400 -- это первое, что нужно смотреть при поломке.
Кириллицу в значение класть нельзя: http-заголовки в requests кодируются
latin-1 и падают с UnicodeEncodeError.

ПОЧЕМУ ДВА ЗАПРОСА. printer/react отдаёт для каждого матча плоскую «главную»
линию (1X2, двойной шанс и по одной центральной форе/тоталу) -- только там
есть исход матча 1X2. ross/data отдаёт лестницы: все форы (tid 448) и все
тоталы (tid 129), но 1X2 в нём нет вовсе. Поэтому берём оба и склеиваем.
Эндпоинт /ajax/line/ross/react_js_line/ отдаёт 500 -- не использовать.

ФОРМАТ ЯЧЕЙКИ. И там и там коэффициент лежит в объекте с полями
{'h': 1.65, 'oddKey': '26638745|9|-1'}. oddKey = 'gid|тип' или 'gid|тип|линия',
линия -- со знаком и со стороны той команды, которой даётся (см. _SEL).
ЛОВУШКА: в тех же массивах лежат ячейки-заголовки вида {'h': '-1'} и
{'h': 'Фора 1'} -- у них h СТРОКА. Строковые h пропускаем всегда, иначе
заголовок линии «-1» превратится в коэффициент.

ИМЕНА КОМАНД. dict.cmd[id] -- русские, dict.eng.cmd[id] -- английские.
Берём английские: norm_team на них попадает точнее (у русских транслитераций
больше вариантов написания). Совпадать должны все 14 команд лиги; если
появилось нераспознанное имя -- это дыра в словаре books/_ALIASES,
матч тихо пропускается, см. вывод __main__ (строка «не распознаны»).

ФОРЫ У ЗЕНИТА АЗИАТСКИЕ. Пара Ф1(0)/Ф2(0) на равном матче котируется
1.85/1.85 -- это возврат при ничьей (европейская фора 0 стоила бы ~2.2).
Поэтому всё уходит через sel_ah; sel_eh не используется -- у Зенита в
росписи вообще нет блока европейских фор. Целые линии (0, -1, 3) -- тоже
азиатские, с возвратом; не расщепляем, отдаём как есть, это дело pricing.

СМЕНА СЕЗОНА / ПОТЕРЯ ЛИГИ. Числовой id лиги (LEAGUE_ID) переживает тур,
но новый сезон Зенит обычно заводит новой записью. Признак поломки --
fetch() вернул пустой список при живом ответе 200. Новый id ищется так:

    python src/books/zenit.py --find-league

то есть GET /ajax/line/left_menu/get (~2.9 МБ, дёргать руками раз в сутки,
не чаще) -> result.dict.league = {id: 'ОАЭ. Про-лига'}.

    PYTHONIOENCODING=utf-8 python src/books/zenit.py
"""
import os
import sys
import time

import requests

# Запуск файла напрямую (python src/books/zenit.py) кладёт в sys.path саму
# папку books, а не src, и пакет books становится невидим. Сборщик импортирует
# модуль нормально, поэтому чинить нужно только сценарий «запустили руками».
if __package__ in (None, ''):
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from books import Quote, norm_team, sel_total, sel_ah, UA  # noqa: E402

KEY = 'zenit'
LEAGUE_ID = 250833          # ОАЭ. Про-лига, см. --find-league
CLIENT_V = '1.72.1'         # версия фронта; сервер её не сверяет, но шлём как браузер

_HOST = 'https://zenit.win'
# length=100 -- с большим запасом: в лиге 7 матчей в туре. Если матчей в
# выдаче меньше, чем в туре, это НЕ обрезка нашим length (проверено all=1,
# length=500, days=30 -- ответ тот же): Зенит открывает тур не целиком,
# ближние матчи появляются в линии раньше дальних.
_LINE = (_HOST + '/ajax/line/printer/react?all=0&onlyview=0&timeline=0'
         '&tournaments_mode=1&sport=1&league={league}&ross=0&lang_id=1'
         '&timezone=3&offset=0&show_from_main=0&client_v={cv}&length=100&sort_mode=2')
_ROSS = (_HOST + '/ajax/line/ross/data/?onlyview=0&gid={gids}&lang_id=1'
         '&client_v={cv}&show_from_main=0')
_MENU = (_HOST + '/ajax/line/left_menu/get?lang_id=1&sort_mode=2&tournaments_mode=1')

_HDRS = {
    'User-Agent': UA,
    'Accept': 'application/json, text/plain, */*',
    'Accept-Language': 'ru-RU,ru;q=0.9,en;q=0.8',
    'Referer': _HOST + '/line/1',
    'X-Requested-With': 'XMLHttpRequest',
    'imprintHash': 'uae-line-reader',   # значение произвольное, важен сам факт
    'frontVersion': CLIENT_V,
}

# Разделы полной росписи, из которых берём цены. Ограничение по tid --
# не косметика: в соседних разделах живут ТАЙМОВЫЕ рынки (tid 137 -- тотал
# 1-го тайма, типы 114/115), и без фильтра по разделу легко утащить тотал
# тайма как тотал матча, если Зенит переиспользует номер типа.
_ROSS_TIDS = ('448',    # Форы (Ф1/Ф2, вся лестница)
              '129',    # Тоталы матча (М/Б, вся лестница)
              '2965')   # Двойной шанс (дубль главной линии, для страховки)

# Тип из oddKey -> канонический ключ исхода без линии.
_MAIN = {'4': '1', '5': 'X', '6': '2',
         '7': '1X', '2': '12', '8': 'X2'}   # да, двойной шанс «12» имеет тип 2

# Тип из oddKey -> как построить ключ исхода с линией.
_SEL = {
    '9':  lambda v: sel_ah(1, v),        # фора команде 1, линия уже со знаком
    '10': lambda v: sel_ah(2, v),        # фора команде 2
    '11': lambda v: sel_total(False, v),  # «М» -- тотал меньше
    '12': lambda v: sel_total(True, v),   # «Б» -- тотал больше
}


def _get(url, tries=3, timeout=45):
    """
    GET + json. Сетевые ошибки НЕ глушим: после последней попытки
    исключение летит наверх, его ловит books.fetch_book.
    """
    last = None
    for i in range(tries):
        try:
            r = requests.get(url, headers=_HDRS, timeout=timeout)
            r.raise_for_status()
            return r.json()
        except Exception as e:      # noqa: BLE001 -- ретраим любую сетевую беду
            last = e
            if i < tries - 1:
                time.sleep(1.5 * (i + 1))
    raise last


def _price(cell):
    """
    Коэффициент из ячейки или None.

    Именно здесь отсекаются ячейки-заголовки: у них h -- строка ('-1', 'Б',
    'Фора 1'). bool отсекаем отдельно, он подкласс int.
    """
    if not isinstance(cell, dict):
        return None
    h = cell.get('h')
    if isinstance(h, bool) or not isinstance(h, (int, float)):
        return None
    p = float(h)
    return p if 1.0 < p < 1000.0 else None


def _sel_of(odd_key):
    """'26638745|9|-1' -> ('26638745', 'AH1-1'); неизвестный рынок -> (gid, None)."""
    parts = str(odd_key).split('|')
    if len(parts) < 2:
        return None, None
    gid, typ = parts[0], parts[1]
    if len(parts) == 2:
        return gid, _MAIN.get(typ)
    if typ not in _SEL:
        return gid, None
    try:
        line = float(parts[2])
    except (TypeError, ValueError):
        return gid, None
    return gid, _SEL[typ](line)


def _walk(node, out):
    """Роспись -- дерево из вложенных ch/data; собираем все ячейки с oddKey."""
    if isinstance(node, dict):
        if 'oddKey' in node:
            out.append(node)
        for v in node.values():
            _walk(v, out)
    elif isinstance(node, list):
        for v in node:
            _walk(v, out)


def _chunks(seq, n):
    for i in range(0, len(seq), n):
        yield seq[i:i + n]


def fetch():
    """-> list[Quote] по лиге ОАЭ. Пустой список = Зенит сейчас не котирует лигу."""
    data = _get(_LINE.format(league=LEAGUE_ID, cv=CLIENT_V))
    # Между турами эндпоинт отвечает 200 с телом null: лиги в меню нет вовсе
    # (резолвер по «ОАЭ» ничего не находит). Это пустая линия, а не ошибка --
    # 12 сентября 2026 адаптер на этом падал с AttributeError.
    if not isinstance(data, dict):
        return []
    games = data.get('games') or {}
    if not games:
        return []
    eng = ((data.get('dict') or {}).get('eng') or {}).get('cmd') or {}
    rus = (data.get('dict') or {}).get('cmd') or {}

    # gid -> (home, away, raw_home, raw_away, kickoff); матчи с неопознанной
    # командой сюда не попадают и дальше просто не существуют
    meta = {}
    for gid, g in games.items():
        if int(g.get('lid') or 0) != LEAGUE_ID:
            continue    # чужая лига в ответе -- страховка, обычно не бывает
        c1, c2 = str(g.get('c1_id')), str(g.get('c2_id'))
        raw_h = eng.get(c1) or rus.get(c1) or ''
        raw_a = eng.get(c2) or rus.get(c2) or ''
        home, away = norm_team(raw_h), norm_team(raw_a)
        if not home or not away:
            continue    # молодёжка/кубок/новая команда -- пропуск молча, см. контракт
        ko = g.get('time')
        meta[str(gid)] = (home, away, raw_h, raw_a,
                          float(ko) if ko else None)
    if not meta:
        return []

    # (gid, sel) -> цена. Главная линия кладётся первой и не перетирается
    # лестницей: в спорной ситуации доверяем тому, что показано в списке.
    prices = {}

    def take(cells):
        for cell in cells:
            p = _price(cell)
            if p is None:
                continue
            gid, sel = _sel_of(cell.get('oddKey'))
            if sel and gid in meta:
                prices.setdefault((gid, sel), p)

    for gid, g in games.items():
        if str(gid) in meta:
            take(g.get('f_l') or [])

    # Полная роспись. Зенит принимает несколько gid через дефис; бьём на
    # пачки, чтобы не упереться в длину URL, если лига разрастётся.
    gids = sorted(meta)
    for pack in _chunks(gids, 10):
        ross = _get(_ROSS.format(gids='-'.join(pack), cv=CLIENT_V))
        for gid, blk in (ross.get('t_b') or {}).items():
            if str(gid) not in meta:
                continue
            tids = ((blk.get('data') or {}).get('data') or {})
            cells = []
            for tid in _ROSS_TIDS:
                if tid in tids:
                    _walk(tids[tid], cells)
            take(cells)

    out = []
    for (gid, sel), price in prices.items():
        home, away, raw_h, raw_a, ko = meta[gid]
        out.append(Quote(book=KEY, home=home, away=away, sel=sel, price=price,
                         kickoff=ko, raw_home=raw_h, raw_away=raw_a))
    return out


def find_league_id(needle='ОАЭ'):
    """
    Переискать id лиги, когда LEAGUE_ID протух (новый сезон).
    Ответ ~2.9 МБ, поэтому руками и редко, из fetch() не вызывается.
    """
    res = _get(_MENU, timeout=120).get('result') or {}
    leagues = ((res.get('dict') or {}).get('league') or {})
    return {int(k): v for k, v in leagues.items() if needle.lower() in str(v).lower()}


if __name__ == '__main__':
    if '--find-league' in sys.argv:
        for lid, name in sorted(find_league_id().items()):
            print(f'{lid}  {name}')
        raise SystemExit

    qs = fetch()
    matches = {}
    for q in qs:
        matches.setdefault((q.home, q.away), []).append(q)
    print(f'zenit: лига {LEAGUE_ID}, матчей {len(matches)}, котировок {len(qs)}')

    order = {'1': 0, 'X': 1, '2': 2, '1X': 3, '12': 4, 'X2': 5}
    for (home, away), qq in sorted(matches.items(),
                                   key=lambda kv: kv[1][0].kickoff or 0):
        q0 = qq[0]
        when = (time.strftime('%d.%m %H:%M', time.localtime(q0.kickoff))
                if q0.kickoff else 'время неизвестно')
        print(f'\n{home} - {away}   ({when})   [{q0.raw_home} - {q0.raw_away}]')
        by = {q.sel: q.price for q in qq}
        line = [f'{s} {by[s]:.2f}' for s in order if s in by]
        print('   исход/ДШ:  ' + ('  '.join(line) if line else '-'))
        for tag, pref in (('форы', 'AH'), ('тоталы', ('O', 'U'))):
            sels = sorted(s for s in by if s.startswith(pref))
            print(f'   {tag} ({len(sels)}): '
                  + '  '.join(f'{s} {by[s]:.2f}' for s in sels[:14]))

    # нераспознанные имена: явный сигнал, что пора чинить books/_ALIASES
    raw = _get(_LINE.format(league=LEAGUE_ID, cv=CLIENT_V))
    eng = ((raw.get('dict') or {}).get('eng') or {}).get('cmd') or {}
    bad = sorted({n for n in eng.values() if not norm_team(n)})
    print('\nне распознаны norm_team: ' + (', '.join(bad) if bad else 'нет'))
