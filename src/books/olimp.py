# -*- coding: utf-8 -*-
"""
Снятие линии OLIMPBET (olimp.bet) по чемпионату ОАЭ.

Публичный JSON API v4, GET, без логина и без кук. Тот же самый, которым
пользуется сайт, поэтому цены совпадают с витриной один в один.

    https://www.olimp.bet/api/v4/<lang>/line/<метод>?vids[]=<id>:

ДВЕ ЛОВУШКИ, на которых всё ломается, если о них не знать:

  1. `vids[]` обязан быть URL-кодирован И заканчиваться ДВОЕТОЧИЕМ ("11906:").
     Без двоеточия сервер отвечает 400 либо пустым массивом. requests кодирует
     сам, двоеточие дописываем руками -- см. _get().
  2. Коэффициент лежит в поле "probability" (строкой!), а не в "odds"/"coef".
     Поля "probability" с вероятностью не имеет ничего общего, это десятичный
     кэф; имя историческое.

Сегмент пути после /v4/ -- это ЯЗЫК ЧИСЛОМ: 0 = ru, 2 = en. Строки "ru"/"en"
дают 500. Берём 0 (русский), потому что английские имена всё равно приезжают
отдельным полем names["2"] в любом языковом варианте.

ДВА ЗАПРОСА, а не один. Почему:
  * competitions-with-events отдаёт список матчей лиги, но всего ~38 исходов
    на матч -- только 1X2, двойной шанс, целые тоталы и целые форы. Азиатских
    линий (четвертных) там нет вообще.
  * line/events отдаёт полную роспись, ~490 исходов на матч, включая
    азиатские тоталы и форы. Именно они нам и нужны: смысл проекта -- брать
    максимум цены по многим конторам, а четвертные линии Olimp котирует
    заметно щедрее целых.
Поэтому: сначала берём id матчей лиги, затем одним запросом (vids[] можно
повторять) тянем полную роспись пачкой.

ЧТО СЛОМАЕТСЯ ПРИ СМЕНЕ СЕЗОНА. LEAGUE_ID жёстко зашит. Когда он протухнет,
fetch() вернёт пустой список молча -- это штатное поведение «контора не
котирует лигу», отличить его от «id умер» можно так:

    python src/books/olimp.py --find-league

Метод пройдёт по sports-with-categories-with-competitions и покажет id по
ТОЧНОМУ совпадению names["2"] == "UAE. Pro League". Точному -- не подстроке:
рядом в справочнике живёт "UAE. Pro League (U23)" (id 8337582), молодёжка,
и подстрочный поиск утащит именно её. countryCode у соревнования равен null,
опираться на него нельзя, только на имя.
"""
import sys
import time

import requests

try:
    from books import Quote, norm_team, sel_total, sel_ah, sel_eh, UA
except ImportError:  # запуск файлом: python src/books/olimp.py
    import os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from books import Quote, norm_team, sel_total, sel_ah, sel_eh, UA

BOOK = 'olimp'
API = 'https://www.olimp.bet/api/v4/0/line'   # 0 = русский, см. модульный docstring
LEAGUE_ID = '11906'                           # ОАЭ. Про-лига, сезон 2026/27
TIMEOUT = 40                                  # ответ честно медленный, 3-6 с -- меньше 30 не ставить
CHUNK = 8                                     # матчей в одном запросе росписи (~1 МБ на 5 матчей)

HDRS = {
    'User-Agent': UA,
    'Accept': 'application/json',
    'Referer': 'https://www.olimp.bet/line/football',
    # gzip requests просит и распаковывает сам; без него ответ втрое толще
}

# ---------------------------------------------------------------------------
#                         КАРТА РЫНКОВ OLIMP -> НАШИ КЛЮЧИ
# ---------------------------------------------------------------------------
# Опознаём рынок по числовому marketId, а НЕ по shortName и не по groupName:
#   * shortName неуникален -- у тотала матча 1.5 и у инд. тотала команды 2
#     он буквально один и тот же ('Тот2Тот2М'). Разберёшь по нему -- склеишь
#     тотал матча с тоталом команды и получишь мусор в консенсусе.
#   * groupName пляшет регистром ('Доп. Тотал' и 'Доп. тотал' в одном ответе).
# marketId дублируется третьим сегментом basketId, сверено на всей линии --
# расхождений нет.
M_1X2 = 1        # Исход матча (основное время)
M_DC = 3         # Двойной шанс
M_HANDICAP = 4   # «Победа с учетом форы» -- целые и половинные линии
M_TOTAL = 5      # Тотал матча -- целые и половинные линии
M_TOTAL_AS = 166  # Азиатские тоталы -- четвертные линии
M_HANDICAP_AS = 168  # Азиатские форы -- четвертные линии

MARKETS = (M_1X2, M_DC, M_HANDICAP, M_TOTAL, M_TOTAL_AS, M_HANDICAP_AS)

# basketId = "<матч>:<вид>:<рынок>:<параметр>:<номер>:NULL:NULL:1".
# «вид» кодирует сторону ставки и нужен там, где номер исхода всегда 1:
#   1 -- многоисходный рынок, сторону задаёт «номер» (1X2, двойной шанс)
#   2 -- «меньше» / «да» / фора (у фор сторону всё равно задаёт «номер»)
#   3 -- «больше» / «нет»
KIND_UNDER, KIND_OVER = '2', '3'

# Порядок исходов в многоисходных рынках -- по «номеру» из basketId.
_1X2 = {'1': '1', '2': 'X', '3': '2'}
_DC = {'1': '1X', '2': '12', '3': 'X2'}

# ПОЧЕМУ ФОРЫ -- ЭТО AH, А НЕ EH. У Olimp в группе «Победа с учетом форы»
# ровно ДВА исхода на линию (Ф1 и Ф2), ничьей нет даже на целых линиях (0, -1,
# -2). Значит при точном попадании счёта в фору ставка возвращается -- это
# азиатская фора с возвратом. Европейская фора трёхисходная и на пуше
# проигрывает; такого рынка Olimp по этой лиге не даёт вовсе, поэтому sel_eh
# здесь не используется (импортируется только ради единой строки импорта
# контракта). Если Olimp когда-нибудь добавит трёхисходную фору -- у неё будет
# свой marketId и «номер» 1/2/3, вот тогда и появится ветка с sel_eh.


def _get(method, vids=(), tries=3):
    """
    GET к API. Сетевые ошибки НЕ глушим -- после ретраев исключение летит
    наверх, его ловит books.fetch_book и отличает поломку от пустой линии.
    """
    # каждый id обязан оканчиваться двоеточием, иначе 400 -- ловушка №1
    params = [('vids[]', f'{v}:') for v in vids]
    last = None
    for i in range(tries):
        try:
            r = requests.get(f'{API}/{method}', params=params,
                             headers=HDRS, timeout=TIMEOUT)
            r.raise_for_status()
            return r.json()
        except Exception as e:
            last = e
            if i == tries - 1:
                raise
            time.sleep(2 * (i + 1))
    raise last


def _price(outcome):
    """Кэф из "probability" (строка). Мусор и заглушки -> None."""
    try:
        p = float(outcome.get('probability'))
    except (TypeError, ValueError):
        return None
    return p if p > 1.0 else None


def _line(outcome):
    """Параметр линии (тотал / фора). Уже подписан со стороны своей команды."""
    try:
        return float(outcome.get('param'))
    except (TypeError, ValueError):
        return None


def _kickoff(payload):
    """startDateTime -- unix-секунды у матча (но миллисекунды у соревнования)."""
    ts = payload.get('startDateTime')
    if not isinstance(ts, (int, float)) or ts <= 0:
        return None
    return float(ts) / 1000.0 if ts > 1e12 else float(ts)


def _teams(payload):
    """
    -> (raw_home, raw_away, home, away) либо None, если матч не наш.

    Английские имена берём из names["2"] ("Al Ain - Al Wasl"): русские
    team1Name/team2Name словарь местами не знает ('Иттихад Кальба' с мягким
    знаком в norm_team не попадает), а английские ложатся все до одного.
    Русские оставлены запасным вариантом на случай, если names["2"] опустеет.
    """
    names = payload.get('names') or {}
    en = (names.get('2') or '').strip()
    raw_h, sep, raw_a = en.partition(' - ')
    if not sep:
        raw_h = (payload.get('team1Name') or '').strip()
        raw_a = (payload.get('team2Name') or '').strip()
    raw_h, raw_a = raw_h.strip(), raw_a.strip()

    home, away = norm_team(raw_h), norm_team(raw_a)
    # None -- НЕ ошибка: в соседних разделах живут молодёжка и кубковые команды
    if not home or not away or home == away:
        return None
    return raw_h, raw_a, home, away


def _selection(outcome):
    """Один исход Olimp -> канонический ключ или None, если рынок нам не нужен."""
    try:
        mid = int(outcome.get('marketId'))
    except (TypeError, ValueError):
        return None
    if mid not in MARKETS:
        return None

    parts = (outcome.get('basketId') or '').split(':')
    if len(parts) < 5:
        return None
    kind, num = parts[1], parts[4]

    if mid == M_1X2:
        return _1X2.get(num)
    if mid == M_DC:
        return _DC.get(num)

    if mid in (M_TOTAL, M_TOTAL_AS):
        line = _line(outcome)
        if line is None or kind not in (KIND_UNDER, KIND_OVER):
            return None
        return sel_total(kind == KIND_OVER, line)

    if mid in (M_HANDICAP, M_HANDICAP_AS):
        line = _line(outcome)
        # «номер» = чьей команде фора; param уже со знаком с её стороны
        # (сверено с текстом в скобках у всех исходов линии -- расхождений нет)
        if line is None or num not in ('1', '2'):
            return None
        return sel_ah(int(num), line)
    return None


def _quotes(payload):
    """Полная роспись одного матча -> список Quote."""
    teams = _teams(payload)
    if not teams:
        return []
    raw_h, raw_a, home, away = teams
    kickoff = _kickoff(payload)

    best = {}
    for o in payload.get('outcomes') or []:
        sel = _selection(o)
        if not sel:
            continue
        price = _price(o)
        if price is None:
            continue
        # Дублей между рынками сейчас нет (целые линии в 4/5, четвертные в
        # 166/168, множества не пересекаются), но если Olimp начнёт дублировать
        # линию в двух группах -- берём лучшую цену, а не последнюю встречную.
        if price > best.get(sel, 0.0):
            best[sel] = price

    return [Quote(book=BOOK, home=home, away=away, sel=sel, price=price,
                  kickoff=kickoff, raw_home=raw_h, raw_away=raw_a,
                  extra={'event_id': str(payload.get('id') or '')})
            for sel, price in best.items()]


def _league_events():
    """Матчи лиги: только id, имена и время. Роспись здесь урезанная."""
    data = _get('competitions-with-events', [LEAGUE_ID])
    out = []
    for block in data or []:
        payload = (block or {}).get('payload') or {}
        for ev in payload.get('events') or []:
            # OPEN -- единственное состояние, которое отдаёт прематч-раздел.
            # Если Olimp заведёт новые имена состояний, адаптер тихо опустеет;
            # тогда просто снять это условие (одна строка).
            if (ev.get('state') or 'OPEN') != 'OPEN':
                continue
            if ev.get('id'):
                out.append(ev)
    return out


def _full_rosters(ids):
    """Полная роспись пачкой: vids[] можно повторять, ~1 МБ на 5 матчей."""
    out = []
    for i in range(0, len(ids), CHUNK):
        for block in _get('events', ids[i:i + CHUNK]) or []:
            payload = (block or {}).get('payload') or {}
            if payload.get('outcomes'):
                out.append(payload)
    return out


def fetch():
    """-> list[Quote] по лиге ОАЭ. Пусто = Olimp сейчас лигу не котирует."""
    events = _league_events()
    if not events:
        return []

    rosters = _full_rosters([str(e['id']) for e in events])
    # Время старта в росписи есть, но в списке лиги оно надёжнее (там же его
    # показывает витрина), поэтому при склейке предпочитаем список.
    starts = {str(e['id']): e.get('startDateTime') for e in events}
    out = []
    for payload in rosters:
        ts = starts.get(str(payload.get('id')))
        if ts:
            payload = dict(payload, startDateTime=ts)
        out.extend(_quotes(payload))
    return out


def find_league_id(needle='UAE. Pro League'):
    """
    Заново найти id лиги, когда LEAGUE_ID протухнет между сезонами.
    Сравнение ТОЧНОЕ: рядом лежит "UAE. Pro League (U23)" -- молодёжка.
    -> список (id, имя, матчей), точное совпадение первым.
    """
    found = []

    def scan(node):
        if isinstance(node, dict):
            names = node.get('names')
            if isinstance(names, dict) and node.get('id') and needle in str(names.get('2', '')):
                found.append((str(node['id']), names.get('2', ''),
                              node.get('eventCount')))
            for v in node.values():
                scan(v)
        elif isinstance(node, list):
            for v in node:
                scan(v)

    scan(_get('sports-with-categories-with-competitions'))
    return sorted(set(found), key=lambda r: (r[1] != needle, r[1]))


if __name__ == '__main__':
    if '--find-league' in sys.argv:
        print('поиск id лиги по справочнику Olimp:')
        for lid, name, cnt in find_league_id():
            mark = '  <-- этот' if name == 'UAE. Pro League' else ''
            print(f'  {lid:<12} {name:<28} матчей {cnt}{mark}')
        sys.exit(0)

    qs = fetch()
    games = {}
    for q in qs:
        games.setdefault((q.home, q.away), []).append(q)

    print(f'{BOOK}: матчей {len(games)}, котировок {len(qs)}')
    for (home, away), lst in sorted(
            games.items(), key=lambda kv: kv[1][0].kickoff or 0):
        q0 = lst[0]
        when = (time.strftime('%Y-%m-%d %H:%M', time.localtime(q0.kickoff))
                if q0.kickoff else '?')
        sels = {q.sel: q.price for q in lst}
        tot = sorted(s for s in sels if s[0] in 'OU')
        ah = sorted(s for s in sels if s.startswith('AH'))
        print(f'\n{home} - {away}   {when}   исходов {len(lst)}')
        print(f'  у конторы: {q0.raw_home} - {q0.raw_away}')
        print('  1X2      ' + '  '.join(f'{s} {sels[s]:.2f}'
                                        for s in ('1', 'X', '2') if s in sels))
        print('  дв.шанс  ' + '  '.join(f'{s} {sels[s]:.2f}'
                                        for s in ('1X', '12', 'X2') if s in sels))
        print(f'  тоталы   {len(tot)} линий: '
              + '  '.join(f'{s} {sels[s]:.2f}' for s in tot[:4]) + ' ...')
        print(f'  форы     {len(ah)} линий: '
              + '  '.join(f'{s} {sels[s]:.2f}' for s in ah[:4]) + ' ...')
