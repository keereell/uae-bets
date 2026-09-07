# -*- coding: utf-8 -*-
"""
Снятие линии Flashscore по чемпионату ОАЭ. Это НЕ контора, а агрегатор:
один запрос отдаёт цены нескольких десятков букмекеров сразу. Ради этого
модуль и написан -- см. рассуждение в src/books/__init__.py: прибыль даёт
не «мягкая контора», а МАКСИМУМ цены по многим конторам, а значит главная
ценность источника -- ширина выборки, а не собственная линия.

Ключ каждой котировки -- 'fs:<имя конторы>' ('fs:1xBet', 'fs:Betfair', ...).
Префикс обязателен: цена, увиденная через Flashscore, -- это НЕ цена, по
которой мы можем поставить (нет счёта, нет гарантии, что линия свежая),
поэтому она не должна случайно смешаться с 'betcity' и прочими из BETTABLE.

ДВА ЗАПРОСА, И ОБА ОБЯЗАТЕЛЬНЫ.

  1) РАСПИСАНИЕ ДНЯ (какие вообще матчи есть и какой у них eventId)
     GET https://global.flashscore.ninja/2/x/feed/f_1_{d}_3_en_1
     d -- смещение дня от сегодня, 0..7. Заголовок x-fsign: SW9D1eZo
     обязателен, без него 401. Ответ -- НЕ json: записи разделены '~',
     поля внутри записи '¬', пара пишется как 'ключ÷значение'.
     Нужные поля: ZA -- заголовок турнира, ZL -- его url;
     AA -- eventId, AD -- unix старта, AE/AF -- имена команд,
     JA/JB -- id участников (home/away), они же придут в котировках.

  2) КОТИРОВКИ ОДНОГО МАТЧА (обычный JSON)
     GET https://global.ds.lsapp.eu/odds/pq_graphql
         ?_hash=oce&eventId={AA}&projectId=2&geoIpCode={CC}&geoIpSubdivisionCode=
     Только User-Agent, ни подписи, ни куки.

ПОЧЕМУ ПЕРЕБИРАЕТСЯ geoIpCode. Flashscore показывает те конторы, которые
разрешены в стране посетителя, и набор различается радикально. Замерено на
матче Al Ain -- Al Wasl: пустой код даёт 6 контор, BR добавляет 26, IT -- 13,
GB -- 6, RO -- 5; итого по списку GEO_CODES набирается ~67 контор с реальными
ценами против 6 «по умолчанию». Один и тот же букмекер под разными кодами
приходит с одним id, поэтому дубли просто отбрасываются (побеждает первый
код из списка -- порядок фиксированный, результат воспроизводим).

ЧТО СЛОМАЕТСЯ ПЕРВЫМ. По убыванию вероятности:

  - _hash=oce. Это persisted query: хэш конкретного текста GraphQL-запроса,
    зашитый во фронтовый бандл. При релизе фронта он меняется, и сервер
    отвечает 404 с телом 'Query not stored'. Модуль это ловит и сам
    перечитывает константу ODDS_EVENT_COMPARISON из /res/_fs/build/detail.*.js
    на зеркале flashscore.co.uk (см. _resolve_hash). Руками то же самое:
    открыть страницу матча, найти в html ссылку на detail.<хэш>.js, скачать,
    поискать 'ODDS_EVENT_COMPARISON='. Мёртвые хэши ope/ope2 и старый фид
    df_od_1_{id} не воскрешать -- они отдают 404 навсегда.

  - x-fsign. Константа фронта, меняется редко, но своим 401 положит шаг 1
    целиком. Нового значения взять неоткуда, кроме как из бандла/девтулзов.

  - Смена сезона. Лига опознаётся по ZL == '/football/united-arab-emirates/
    uae-league/'. Это url раздела, он переживает смену сезона; заголовок ZA
    ('UNITED ARAB EMIRATES: UAE League') используется только как запасной
    признак. Брать по стране НЕЛЬЗЯ: в том же ZY='United Arab Emirates'
    лежат 'Pro League U23' (молодёжка) и 'Division 1' (второй дивизион) --
    другие турниры с похожими именами команд.

  - Новая команда, которой нет в _ALIASES: norm_team вернёт None, матч молча
    выпадет. Видно в выводе __main__ строкой 'НЕ РАСПОЗНАНЫ'; лечится
    синонимом в src/books/__init__.py, не здесь.

СЕМАНТИКА РЫНКОВ (проверена арифметикой на живой линии, а не по документации,
которой нет). Берём только bettingScope == 'FULL_TIME':

  HOME_DRAW_AWAY   eventParticipantId == JA -> '1', == JB -> '2', null -> 'X'.
  DOUBLE_CHANCE    в eventParticipantId стоит команда, входящая в ОБА исхода:
                   JA -> '1X', JB -> 'X2', null (исход без ничьей) -> '12'.
                   Сверено по 1X2 того же букмекера: при 1X2 = 2.15/2.40/4.49
                   ДШ = 1.13(JA)/1.43(null)/1.55(JB), что даёт ровно
                   1X=1.13, 12=1.45, X2=1.56 -- совпадение однозначное.
  OVER_UNDER       handicap.value -- линия, selection -- OVER/UNDER.
  ASIAN_HANDICAP   handicap.value УЖЕ подписан со стороны своего участника:
                   запись (JA, +0.75) -- это фора хозяевам +0.75. Пара к ней
                   лежит отдельной записью (JB, -0.75), а не в этой же.
                   Проверено маржой: 1/1.10 + 1/5.05 = 1.107 и
                   1/1.85 + 1/1.77 = 1.106 -- обе пары дают одну и ту же
                   книгу, значит группировка по знаку верна.
                   Четверти (0.25/0.75) отдаются как есть -- расщепляет pricing.
  EUROPEAN_HANDICAP то же самое, целые линии. Записи с eventParticipantId
                   null -- это «ничья с форой» (например, хозяева проиграли
                   ровно 1 мяч); канонического ключа для такого исхода в
                   контракте нет, поэтому они отбрасываются.

Прочие типы (BOTH_TEAMS_TO_SCORE, CORRECT_SCORE, HALF_FULL_TIME, ODD_OR_EVEN,
DRAW_NO_BET) в контракт не укладываются и игнорируются молча.

ЧЕГО ЖДАТЬ ОТ ОБЪЁМА. Часть контор приходит в odds[] с ПУСТЫМ внутренним
списком цен: блок есть, чисел нет. На замере это bet365, bwin, Unibet,
Sportingbet, Netwin, Eurobet.it, BetssonIT и ещё несколько -- то есть из ~67
контор с блоками реальные числа дают ~54. Это не поломка парсера, а политика
источника, чинить нечего. Матч без цен вообще (odds[] пустой у всех контор)
-- тоже норма: линию на ближайший тур конторы открывают не одновременно.

    PYTHONIOENCODING=utf-8 python src/books/flashscore.py
"""
import os
import re
import sys
import time
import datetime as dt

import requests

try:
    from books import Quote, norm_team, sel_total, sel_ah, sel_eh, UA
except ImportError:  # прямой запуск файла: python src/books/flashscore.py
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from books import Quote, norm_team, sel_total, sel_ah, sel_eh, UA

BOOK_PREFIX = 'fs:'

FEED = 'https://global.flashscore.ninja/2/x/feed/f_1_{d}_3_en_1'
FSIGN = 'SW9D1eZo'          # подпись фронта; без неё фид отвечает 401
ODDS = 'https://global.ds.lsapp.eu/odds/pq_graphql'
ODDS_HASH = 'oce'           # persisted query ODDS_EVENT_COMPARISON, см. шапку
MIRROR = 'https://www.flashscore.co.uk'   # оттуда перечитываем хэш, если протух

LEAGUE_URL = '/football/united-arab-emirates/uae-league/'
LEAGUE_TITLE = 'UNITED ARAB EMIRATES: UAE League'

# Сколько дней расписания просматривать вперёд. Тур ОАЭ укладывается в 3-4 дня,
# запас взят на переносы и на сдвоенные туры.
DAYS = 8

# Порядок важен: при совпадении id букмекера побеждает первый код, поэтому
# от перестановки списка меняется, чья именно копия цены попадёт в выдачу.
# Пустой код = «как видит аноним»; дальше страны, добавляющие больше всего
# новых контор (BR даёт +26, IT +13).
GEO_CODES = ['', 'GB', 'DE', 'IT', 'ES', 'BR', 'RU', 'CZ', 'PT', 'GR', 'RO', 'AT']

TIMEOUT = 30
TRIES = 3
PAUSE = 0.12        # межзапросная пауза: запросов под сотню, спешить некуда

FEED_HDRS = {'User-Agent': UA, 'x-fsign': FSIGN,
             'Referer': 'https://www.flashscore.com/',
             'Accept': '*/*'}
ODDS_HDRS = {'User-Agent': UA, 'Accept': 'application/json'}

# Рынки, которые вообще умеем раскладывать в канонические ключи исходов.
WANTED = ('HOME_DRAW_AWAY', 'DOUBLE_CHANCE', 'OVER_UNDER',
          'ASIAN_HANDICAP', 'EUROPEAN_HANDICAP')


class HashExpired(Exception):
    """_hash протух: сервер ответил 404 'Query not stored'."""


# ---------------------------------------------------------------------------
#                                  СЕТЬ
# ---------------------------------------------------------------------------
def _get(session, url, headers, params=None, tries=TRIES):
    """
    GET с ретраями. Сетевую ошибку НЕ глотаем: после последней попытки
    исключение летит наверх, там его ловит fetch_book и отличает
    «контора не котирует лигу» от «мы сломались».
    """
    for i in range(tries):
        try:
            r = session.get(url, headers=headers, params=params, timeout=TIMEOUT)
            if r.status_code == 404 and 'Query not stored' in r.text:
                raise HashExpired(r.text.strip()[:80])
            r.raise_for_status()
            return r
        except HashExpired:
            raise                      # ретраить бессмысленно, лечится иначе
        except Exception:
            if i == tries - 1:
                raise
            time.sleep(1.5 * (i + 1))


def _resolve_hash(session, event_id):
    """
    Перечитать persisted-query хэш из фронтового бандла -- на случай, когда
    зашитый ODDS_HASH протух. Путь: страница КОНКРЕТНОГО матча (только на ней
    подключается detail-бандл, в разделе /football/ его нет) -> ссылка вида
    /res/_fs/build/detail.<версия>.js -> константа ODDS_EVENT_COMPARISON="oce".
    Зеркало .co.uk выбрано потому, что отдаёт html без гео-редиректов.
    """
    html = _get(session, f'{MIRROR}/match/football/{event_id}/',
                {'User-Agent': UA}).text
    for path in dict.fromkeys(re.findall(r'/res/_fs/build/[\w.\-]*detail[\w.\-]*\.js', html)):
        js = _get(session, MIRROR + path, {'User-Agent': UA}).text
        m = re.search(r'ODDS_EVENT_COMPARISON\s*=\s*"([a-z0-9]+)"', js)
        if m:
            return m.group(1)
    raise RuntimeError('не нашли ODDS_EVENT_COMPARISON в бандле detail.*.js')


# ---------------------------------------------------------------------------
#                        ШАГ 1: РАСПИСАНИЕ И МАТЧИ ЛИГИ
# ---------------------------------------------------------------------------
def _records(text):
    """
    Разбор фида. Формат самодельный: '~' режет записи, '¬' -- поля,
    '÷' отделяет ключ от значения. Записи идут потоком: сначала запись
    турнира (в ней есть ZA), потом его матчи (в них есть AA), и так далее.
    """
    for rec in text.split('~'):
        f = {}
        for part in rec.split('¬'):
            k, sep, v = part.partition('÷')
            if sep:
                f[k] = v
        if f:
            yield f


def _events(session):
    """
    -> список сырых матчей лиги ОАЭ на ближайшие DAYS дней.
    Нормализацию имён здесь НЕ делаем: она нужна и fetch(), и отчёту в
    __main__ (там по нераспознанным именам видно дыры в словаре).
    """
    out, seen = [], set()
    league = None
    for d in range(DAYS):
        text = _get(session, FEED.format(d=d), FEED_HDRS).text
        for f in _records(text):
            if 'ZA' in f:
                # начался новый турнир -- запоминаем, наш он или нет
                league = (f.get('ZL') == LEAGUE_URL or f.get('ZA') == LEAGUE_TITLE)
                continue
            if not league or 'AA' not in f or f['AA'] in seen:
                continue
            seen.add(f['AA'])
            out.append(dict(
                id=f['AA'],
                kickoff=int(f['AD']) if f.get('AD', '').isdigit() else None,
                raw_home=f.get('AE') or '',
                raw_away=f.get('AF') or '',
                home_id=f.get('JA') or '',
                away_id=f.get('JB') or '',
            ))
        time.sleep(PAUSE)
    return sorted(out, key=lambda e: e['kickoff'] or 0)


# ---------------------------------------------------------------------------
#                        ШАГ 2: КОТИРОВКИ ПО ОДНОМУ МАТЧУ
# ---------------------------------------------------------------------------
def _sel(item, btype, home_id, away_id):
    """Одна запись котировки -> канонический ключ исхода, либо None."""
    pid = item.get('eventParticipantId')
    side = 1 if pid == home_id else (2 if pid == away_id else None)
    hcp = (item.get('handicap') or {}).get('value')

    if btype == 'HOME_DRAW_AWAY':
        return {1: '1', 2: '2', None: 'X'}[side]

    if btype == 'DOUBLE_CHANCE':
        # в pid -- команда, входящая в оба исхода; null -- «без ничьей»
        return {1: '1X', 2: 'X2', None: '12'}[side]

    if btype == 'OVER_UNDER':
        if hcp is None:
            return None
        s = item.get('selection')
        if s not in ('OVER', 'UNDER'):
            return None
        return sel_total(s == 'OVER', float(hcp))

    if hcp is None or side is None:
        # форы без участника -- это «ничья с форой» у EUROPEAN_HANDICAP,
        # канонического ключа для неё нет
        return None
    line = float(hcp)
    return sel_ah(side, line) if btype == 'ASIAN_HANDICAP' else sel_eh(side, line)


def _event_quotes(session, ev, home, away, hash_box):
    """
    Все котировки одного матча по всем гео-кодам.
    hash_box -- однопозиционный список с текущим _hash: при протухании
    хэш перерезолвится один раз и переиспользуется остальными матчами
    (глобального состояния между вызовами fetch() при этом не возникает).
    """
    quotes, names, retried = {}, {}, False
    for cc in GEO_CODES:
        params = dict(_hash=hash_box[0], eventId=ev['id'], projectId=2,
                      geoIpCode=cc, geoIpSubdivisionCode='')
        try:
            r = _get(session, ODDS, ODDS_HDRS, params=params)
        except HashExpired:
            if retried:
                raise
            retried = True
            hash_box[0] = _resolve_hash(session, ev['id'])
            params['_hash'] = hash_box[0]
            r = _get(session, ODDS, ODDS_HDRS, params=params)

        root = ((r.json().get('data') or {}).get('findOddsByEventId')) or {}
        # словарь id->имя конторы копится по всем гео: в odds[] имени нет,
        # только bookmakerId, а settings[] в каждом ответе своя
        for b in (root.get('settings') or {}).get('bookmakers') or []:
            bm = b.get('bookmaker') or {}
            if bm.get('id') is not None and bm.get('name'):
                names[bm['id']] = bm['name']

        for block in root.get('odds') or []:
            if block.get('bettingScope') != 'FULL_TIME':
                continue
            btype = block.get('bettingType')
            if btype not in WANTED:
                continue
            name = names.get(block.get('bookmakerId'))
            if not name:
                continue          # контора есть в odds[], но не описана -- пропуск
            for item in block.get('odds') or []:
                if not item.get('active'):
                    continue      # снятый с продажи исход
                try:
                    price = float(item.get('value'))
                except (TypeError, ValueError):
                    continue
                if price <= 1.0:
                    # 1.00 и ниже -- заглушка «ставка без выигрыша»; в консенсусе
                    # такая цена означала бы вероятность 1.0 и всё бы испортила
                    continue
                try:
                    sel = _sel(item, btype, ev['home_id'], ev['away_id'])
                except (TypeError, ValueError):
                    continue
                if not sel:
                    continue
                key = (name, sel)
                if key in quotes:
                    continue      # тот же букмекер под другим гео -- дубль
                op = item.get('opening')
                quotes[key] = Quote(
                    book=BOOK_PREFIX + name,
                    home=home, away=away, sel=sel, price=price,
                    kickoff=ev['kickoff'],
                    raw_home=ev['raw_home'], raw_away=ev['raw_away'],
                    extra=dict(event_id=ev['id'], geo=cc or '-',
                               bookmaker_id=block.get('bookmakerId'),
                               opening=float(op) if op else None),
                )
        time.sleep(PAUSE)
    return list(quotes.values())


# ---------------------------------------------------------------------------
#                              ТОЧКА ВХОДА
# ---------------------------------------------------------------------------
def fetch():
    """-> list[Quote] по всем матчам лиги ОАЭ, что есть в расписании."""
    out = []
    hash_box = [ODDS_HASH]
    with requests.Session() as session:
        for ev in _events(session):
            if not (ev['home_id'] and ev['away_id']):
                # без JA/JB не отличить хозяев от гостей, а в котировках сторона
                # задана только id участника: молча пропускаем, иначе весь 1X2
                # матча уехал бы в 'X', а форы исчезли бы. На живом фиде такого
                # не встречалось, страховка от кривой записи.
                continue
            home = norm_team(ev['raw_home'])
            away = norm_team(ev['raw_away'])
            if not home or not away:
                continue          # чужая команда в разделе -- матч не наш, молчим
            out.extend(_event_quotes(session, ev, home, away, hash_box))
    return out


if __name__ == '__main__':
    quotes = fetch()

    games = {}
    for q in quotes:
        games.setdefault((q.home, q.away), []).append(q)
    books = sorted({q.book for q in quotes})

    # Диагностика по расписанию: отдельно нераспознанные имена (дыры в
    # _ALIASES, из-за них матч тихо выпадает) и отдельно матчи, которые
    # распознаны, но цен по ним нет ни у одной конторы.
    with requests.Session() as _s:
        raw_events = _events(_s)
    unknown, dry = [], []
    for e in raw_events:
        pair = norm_team(e['raw_home']), norm_team(e['raw_away'])
        if not (pair[0] and pair[1]):
            unknown.append(f"{e['raw_home']} — {e['raw_away']}")
        elif pair not in games:
            dry.append(f"{pair[0]} — {pair[1]}")

    print(f'Flashscore: матчей в расписании {len(raw_events)}, '
          f'с ценами {len(games)}, котировок {len(quotes)}, '
          f'контор внутри {len(books)}')
    if unknown:
        print('НЕ РАСПОЗНАНЫ (нет в _ALIASES, матч пропущен): ' + '; '.join(unknown))
    if dry:
        print('БЕЗ ЦЕН (линия ещё не открыта, у всех контор odds[] пуст): '
              + '; '.join(dry))
    print('конторы: ' + ', '.join(b[len(BOOK_PREFIX):] for b in books))

    def _fam(s):
        if s in ('1', 'X', '2'):
            return '1X2'
        if s in ('1X', '12', 'X2'):
            return 'ДШ'
        if s[0] in 'OU':
            return 'тотал'
        return 'фора'

    for (home, away), qs in sorted(games.items(),
                                   key=lambda kv: kv[1][0].kickoff or 0):
        ko = qs[0].kickoff
        when = dt.datetime.fromtimestamp(ko).strftime('%Y-%m-%d %H:%M') if ko else '?'
        by = {}
        for q in qs:
            by.setdefault(_fam(q.sel), set()).add(q.sel)
        counts = ', '.join(f'{k} {len(v)} линий' for k, v in sorted(by.items()))
        n_b = len({q.book for q in qs})
        print(f'\n{home} — {away}   {when}   [{qs[0].raw_home} — {qs[0].raw_away}]')
        print(f'  котировок {len(qs)} от {n_b} контор; {counts}')
        # лучшая цена по каждому из главных исходов -- ради этого агрегатор и нужен
        for sel in ('1', 'X', '2', 'O2.5', 'U2.5', 'AH1-0.5', 'AH2+0.5'):
            same = [q for q in qs if q.sel == sel]
            if not same:
                continue
            best = max(same, key=lambda q: q.price)
            worst = min(same, key=lambda q: q.price)
            print(f'    {sel:<8} макс {best.price:>6.2f} ({best.book[len(BOOK_PREFIX):]})'
                  f'   мин {worst.price:>6.2f}   контор {len(same)}')
