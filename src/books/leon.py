# -*- coding: utf-8 -*-
"""
Снятие линии Leon (leon.ru) по чемпионату ОАЭ.

ПОЧЕМУ ИМЕННО ЭТИ ЭНДПОИНТЫ. У Leon публичный JSON REST, тот же, что дёргает
их собственный фронт: ни логина, ни куки, ни подписи запроса. Корень leon.ru
на голый GET отвечает 406 -- это защита от скрапинга HTML, на /api-2/ она не
распространяется, так что HTML нам не нужен вообще.

Два запроса, и оба обязательны:

  1) СПИСОК МАТЧЕЙ ЛИГИ
     GET /api-2/betline/events/all?ctag=ru-RU&league_id=<id>&hideClosed=true
     Отдаёт events[] с id, nameDefault ("Home - Away", АНГЛИЙСКИЕ имена),
     kickoff (МИЛЛИсекунды), competitors[]{name, homeAway} и всего ~9 рынков
     на матч -- только главные линии.

  2) ПОЛНАЯ ЛИНИЯ ОДНОГО МАТЧА
     GET /api-2/betline/event/all?ctag=ru-RU&eventId=<id>
     ~75 рынков: ВСЕ линии тоталов (1.5 ... 4.5 плюс азиатские четверти) и
     ВСЕ линии фор. Ради них второй запрос и делается: агрегатору нужна не
     главная линия, а максимум цены по всей сетке, иначе сравнивать не с чем.

ID ЛИГИ И СМЕНА СЕЗОНА. league_id=1970324836975631 -- у Leon это "Высшая Лига"
(url arabian-gulf-league) в регионе с family=="AE". При смене сезона Leon
обычно оставляет id прежним, но не гарантирует. Поэтому если по зашитому id
матчей ноль -- модуль сам перерезолвит id через
     GET /api-2/betline/sports?ctag=ru-RU&flags=urlv2
(список видов спорта -> regions[] с family=="AE" -> leagues[] с
url=="arabian-gulf-league") и повторит запрос. Взять id руками можно там же.
Внимание: в том же регионе лежит "Про Лига U23" (url u23-pro-league) --
молодёжка, её брать нельзя, поэтому фильтр именно по url лиги, а не по имени.

КАКИЕ РЫНКИ БЕРЁМ (ctag=ru-RU, поэтому имена русские):
     'Исход 1Х2 (основное время)'  -> 1 / X / 2
     'Двойной исход'               -> 1X / 12 / X2
     'Тотал'                       -> целые и половинные линии
     'Азиатский тотал'             -> четвертные линии (2.25, 2.75, ...)
     'Фора'                        -> целые и половинные линии
     'Азиатская фора'              -> четвертные линии
Всё, где в имени есть "тайм", "хозяев", "гостей" -- это сегменты матча и
командные тоталы, они в общий контракт не укладываются и отбрасываются.

ПОЧЕМУ ВСЕ ФОРЫ УХОДЯТ В sel_ah, А НЕ В sel_eh. Проверено по живой линии:
у Leon КАЖДЫЙ рынок с typeTag=="HANDICAP" имеет ровно два исхода (HOME/AWAY),
третьего (ничья с форой) нет ни на одной линии, включая целые (-1, -2).
Два исхода на целой линии = возврат при точном попадании, то есть азиатская
семантика. Европейской формы (три исхода) Leon по этой лиге не котирует.
На случай, если появится, оставлена ветка: рынок с исходом DRAW уйдёт в
sel_eh. Расщеплять четверти (0.25/0.75) здесь нельзя -- это делает pricing.

ЧТО СЛОМАЕТСЯ ПЕРВЫМ. Порядок хрупкости, от самого вероятного:
  - переименование рынков ("Азиатский тотал" -> что-то ещё). Страховка:
    рынок опознаётся по marketTypeId, имя -- только запасной путь;
  - смена marketTypeId. Страховка симметричная: имя из белого списка;
  - смена league_id (новый сезон) -- лечится авторезолвом, см. выше;
  - новая команда в лиге, которой нет в _ALIASES: norm_team вернёт None и
    матч молча выпадет. Это видно по строке "не распознаны" в выводе ниже --
    лечится добавлением синонима в src/books/__init__.py, НЕ здесь.

    PYTHONIOENCODING=utf-8 python src/books/leon.py
"""
import json
import requests
import os
import sys
import time
import urllib.request
import datetime as dt

try:
    from books import Quote, norm_team, sel_total, sel_ah, sel_eh, UA
except ImportError:  # прямой запуск файла: python src/books/leon.py
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from books import Quote, norm_team, sel_total, sel_ah, sel_eh, UA

BOOK = 'leon'

API = 'https://leon.ru/api-2/betline'
# flags -- набор фич фронта; без них ответ беднее рынками, копируются как есть
FLAGS = 'reg,urlv2,mm2,rrc2,nodup'
LEAGUE_ID = 1970324836975631        # ОАЭ, "Высшая Лига" / arabian-gulf-league
LEAGUE_URL = 'arabian-gulf-league'  # опознавательный знак при авторезолве
REGION_FAMILY = 'AE'

# UA не обязателен (API отдаёт и без заголовков), но пусть контора видит
# осмысленного клиента -- дешевле, чем потом разбираться с внезапной 403.
HDRS = {'User-Agent': UA, 'Accept': 'application/json'}

# marketTypeId -> наша семья исходов. Основной способ опознания рынка:
# числа стабильнее локализованных имён.
MT_1X2 = 1970324836974645
MT_DC = 1970324836974649
MT_TOTAL = 1970324836974992      # 'Тотал'            -- целые/половинные
MT_TOTAL_ASIAN = 1970324836974991  # 'Азиатский тотал' -- четверти
MT_HCP = 1970324836975100        # 'Фора'             -- целые/половинные
MT_HCP_ASIAN = 1970324836974637  # 'Азиатская фора'   -- четверти

_BY_TYPE_ID = {
    MT_1X2: '1x2', MT_DC: 'dc',
    MT_TOTAL: 'total', MT_TOTAL_ASIAN: 'total',
    MT_HCP: 'hcp', MT_HCP_ASIAN: 'hcp',
}
# запасной путь опознания -- по точному имени рынка при ctag=ru-RU
_BY_NAME = {
    'Исход 1Х2 (основное время)': '1x2',
    'Двойной исход': 'dc',
    'Тотал': 'total',
    'Азиатский тотал': 'total',
    'Фора': 'hcp',
    'Азиатская фора': 'hcp',
}
# слова-маркеры сегментов матча и командных тоталов: такие рынки нам не нужны
# ни при каком способе опознания (страховка от коллизии id при смене сезона)
_REJECT = ('тайм', 'хозяев', 'гостей', 'овертайм', 'пенальти')

# тег исхода -> ключ двойного шанса
_DC_TAGS = {'HOMEDRAW': '1X', 'HOMEAWAY': '12', 'DRAWAWAY': 'X2'}
_1X2_TAGS = {'HOME': '1', 'DRAW': 'X', 'AWAY': '2'}


# ---------------------------------------------------------------------------
#                                   СЕТЬ
# ---------------------------------------------------------------------------
# Один Session на процесс. Leon отвечает 307 НА САМОГО СЕБЯ и выставляет
# куки spid/spsc: клиент без cookie-jar (urllib) уходит в бесконечный
# редирект, с cookie-jar второй запрос проходит. 11-12 сентября 2026
# адаптер из-за этого выпадал целиком. Session заодно хранит куки между
# проходами опроса -- рукопожатие делается один раз за прогон.
_SESSION = None


def _session():
    global _SESSION
    if _SESSION is None:
        _SESSION = requests.Session()
        _SESSION.headers.update(HDRS)
    return _SESSION


def _get(url, tries=3, timeout=30):
    """GET + json через Session с cookie-jar. Сетевые ошибки не глушим."""
    last = None
    for i in range(tries):
        try:
            r = _session().get(url, timeout=timeout, allow_redirects=True)
            r.raise_for_status()
            return r.json()
        except Exception as e:      # noqa: BLE001 -- ретраим любую сетевую беду
            last = e
            if i < tries - 1:
                time.sleep(1.5 * (i + 1))
    raise last

def _events_url(league_id):
    return (f'{API}/events/all?ctag=ru-RU&league_id={league_id}'
            f'&hideClosed=true&flags={FLAGS}')


def _event_url(event_id):
    return f'{API}/event/all?ctag=ru-RU&eventId={event_id}&flags={FLAGS}'


def resolve_league_id():
    """
    Перерезолвить id лиги ОАЭ, если зашитый перестал отдавать матчи.
    Ищем регион с family=='AE' и лигу с url=='arabian-gulf-league'
    (по имени искать нельзя: рядом лежит молодёжная 'Про Лига U23').
    Возвращает int или None.
    """
    data = _get(f'{API}/sports?ctag=ru-RU&flags=urlv2', timeout=60)
    sports = data if isinstance(data, list) else (data.get('sports') or [])
    for sport in sports:
        for region in sport.get('regions') or []:
            if region.get('family') != REGION_FAMILY:
                continue
            for league in region.get('leagues') or []:
                if league.get('url') == LEAGUE_URL:
                    return league.get('id')
    return None


# ---------------------------------------------------------------------------
#                                  РАЗБОР
# ---------------------------------------------------------------------------
def _num(v):
    """'+1.5' / '3' / 3.0 -> float, иначе None. Leon шлёт линию строкой."""
    if v is None:
        return None
    try:
        return float(str(v).strip().replace(',', '.'))
    except ValueError:
        return None


def _market_family(market):
    """'1x2' | 'dc' | 'total' | 'hcp' | None -- что это за рынок."""
    name = market.get('name') or ''
    low = name.lower()
    if any(w in low for w in _REJECT):
        return None
    fam = _BY_TYPE_ID.get(market.get('marketTypeId'))
    if fam is None:
        fam = _BY_NAME.get(name)
    if fam is None:
        return None
    # typeTag -- третья независимая проверка: тотал обязан быть TOTAL и т.д.
    tag = market.get('typeTag')
    if fam == 'total' and tag != 'TOTAL':
        return None
    if fam == 'hcp' and tag != 'HANDICAP':
        return None
    return fam


def _teams(event):
    """
    -> (home_canon, away_canon, raw_home, raw_away) либо None, если хотя бы
    одна команда не опознана.

    Ориентация (кто хозяин) берётся из competitors[].homeAway -- это
    единственное поле, где Leon говорит об этом явно. Имена для norm_team
    пробуем в порядке: английское из nameDefault ("Home - Away"), затем
    русское из competitors[].name -- в словаре есть и те, и другие, но
    английские ближе к канону и понятнее в логах отладки.
    """
    ru = {}
    for c in event.get('competitors') or []:
        side = c.get('homeAway')
        if side in ('HOME', 'AWAY') and c.get('name'):
            ru[side] = c['name']

    en = {}
    parts = [p.strip() for p in (event.get('nameDefault') or '').split(' - ')]
    if len(parts) == 2 and all(parts):
        en['HOME'], en['AWAY'] = parts

    out = []
    for side in ('HOME', 'AWAY'):
        raw = en.get(side) or ru.get(side)
        if not raw:
            return None
        canon = norm_team(en.get(side)) or norm_team(ru.get(side))
        if not canon:
            # молодёжка/кубковая команда/новичок лиги -- матч пропускаем молча,
            # угадывать имя запрещено контрактом
            return None
        out += [canon, raw]
    return out[0], out[2], out[1], out[3]


def _selections(market, family):
    """
    Рынок -> [(sel, price), ...]. Пустой список, если рынок нам не подошёл.
    """
    runners = [r for r in (market.get('runners') or [])
               if r.get('open', True) and _num(r.get('price'))]
    if not runners:
        return []
    mkt_line = _num(market.get('handicap'))
    if mkt_line is None:
        spec = market.get('specifiers') or {}
        mkt_line = _num(spec.get('total') if family == 'total' else spec.get('hcp'))

    # у форы с тремя исходами (есть ничья) семантика европейская -- сейчас
    # Leon такого не котирует, ветка на будущее
    eur_hcp = family == 'hcp' and any('DRAW' in (r.get('tags') or []) for r in runners)

    out = []
    for r in runners:
        price = _num(r.get('price'))
        tags = r.get('tags') or []
        tag = tags[0] if tags else ''

        if family == '1x2':
            sel = _1X2_TAGS.get(tag)
        elif family == 'dc':
            sel = _DC_TAGS.get(tag)
        elif family == 'total':
            line = _num(r.get('handicap'))
            if line is None:
                line = mkt_line
            if line is None or tag not in ('OVER', 'UNDER'):
                continue
            sel = sel_total(tag == 'OVER', line)
        elif family == 'hcp':
            if tag not in ('HOME', 'AWAY'):
                continue          # ничья с форой в наш контракт не входит
            # у раннера линия уже со ЗНАКОМ и со стороны своей команды:
            # '1 (-1)' -> '-1', '2 (+1)' -> '+1'. Если её нет -- берём линию
            # рынка (она всегда от хозяев) и переворачиваем знак для гостей.
            line = _num(r.get('handicap'))
            if line is None:
                if mkt_line is None:
                    continue
                line = mkt_line if tag == 'HOME' else -mkt_line
            team = 1 if tag == 'HOME' else 2
            sel = sel_eh(team, line) if eur_hcp else sel_ah(team, line)
        else:
            sel = None

        if sel and price and price > 1.0:
            out.append((sel, price))
    return out


def _event_quotes(event, markets):
    """Одно событие + его рынки -> список Quote. Неопознанный матч -> []."""
    teams = _teams(event)
    if not teams:
        return []
    home, away, raw_home, raw_away = teams

    ko = event.get('kickoff')
    kickoff = ko / 1000.0 if isinstance(ko, (int, float)) and ko else None
    # hideClosed=true прячет только закрытые РЫНКИ; начавшийся матч остаётся
    # с открытыми live-исходами. 10 сентября 2026 именно они ушли в консенсус
    # как «предматчевые» и породили 27 ложных сигналов. Общая отсечка есть
    # в books.fetch_all, здесь -- страховка у источника.
    if kickoff and kickoff <= time.time() + 60:
        return []

    # по одному исходу может прийти несколько цен (одна линия разложена на
    # 'Тотал' и 'Азиатский тотал'); для агрегатора max -- единственный
    # осмысленный выбор, он и так берёт максимум по конторам
    best = {}
    for m in markets or []:
        if not m.get('open', True):
            continue
        fam = _market_family(m)
        if not fam:
            continue
        for sel, price in _selections(m, fam):
            if price > best.get(sel, 0.0):
                best[sel] = price

    return [Quote(book=BOOK, home=home, away=away, sel=sel, price=price,
                  kickoff=kickoff, raw_home=raw_home, raw_away=raw_away,
                  extra={'event_id': event.get('id')})
            for sel, price in sorted(best.items())]


# ---------------------------------------------------------------------------
#                                 ТОЧКА ВХОДА
# ---------------------------------------------------------------------------
def fetch():
    """Вся линия Leon по чемпионату ОАЭ -> list[Quote]."""
    league_id = LEAGUE_ID
    data = _get(_events_url(league_id))
    events = data.get('events') or []

    # пусто может быть по двум причинам: пауза между турами (нормально) или
    # сменился league_id (новый сезон). Второе лечится авторезолвом.
    if not events:
        fresh = resolve_league_id()
        if fresh and fresh != league_id:
            events = (_get(_events_url(fresh)).get('events') or [])

    out = []
    for ev in events:
        ev_id = ev.get('id')
        if not ev_id:
            continue
        # ранняя отсечка: если команды не опознаны, полную линию не тянем
        if not _teams(ev):
            continue
        full = _get(_event_url(ev_id))
        # markets из ответа по событию -- те самые ~75 рынков со всеми
        # линиями; если их вдруг нет, довольствуемся главными из списка
        markets = full.get('markets') or ev.get('markets')
        out.extend(_event_quotes(ev, markets))
    return out


if __name__ == '__main__':
    quotes = fetch()

    # что не распознал norm_team -- отдельно: это дыры в словаре синонимов,
    # из-за них матч тихо выпадает из линии
    raw = _get(_events_url(LEAGUE_ID))
    unknown = []
    for ev in raw.get('events') or []:
        if not _teams(ev):
            unknown.append(ev.get('nameDefault') or ev.get('name'))

    games = {}
    for q in quotes:
        games.setdefault((q.home, q.away), []).append(q)

    print(f'Leon: матчей {len(games)}, котировок {len(quotes)}')
    if unknown:
        print('НЕ РАСПОЗНАНЫ (нет в _ALIASES, матч пропущен): '
              + '; '.join(unknown))

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
            by.setdefault(_fam(q.sel), []).append(q)
        counts = ', '.join(f'{k} {len(v)}' for k, v in sorted(by.items()))
        print(f'\n{home} — {away}   {when}   [{qs[0].raw_home} — {qs[0].raw_away}]')
        print(f'  исходов {len(qs)}: {counts}')
        for q in sorted(qs, key=lambda x: x.sel):
            print(f'    {q.sel:<10} {q.price:>7.2f}')
