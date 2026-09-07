# -*- coding: utf-8 -*-
"""
Снятие линии Marathonbet по чемпионату ОАЭ.

ПОЧЕМУ HTML, А НЕ API. Публичного JSON линии у Марафона нет: сайт рисует
купон на сервере и обновляет ячейки точечными ajax-патчами по
data-mutable-id. Поэтому парсим страницу. Плюс в том, что цена лежит прямо
в атрибуте data-selection-price, а не в тексте, -- разметку можно менять,
пока живы атрибуты, парсер не заметит.

ПОЧЕМУ .ru И ПРЕФИКС /su/. marathonbet.com из этого окружения не отвечает,
отвечает только российское зеркало. /su/ -- код локали (русский интерфейс),
без него будет редирект. Заголовки -- обычный браузерный UA, cookies не
нужны, отдаётся с Cache-Control: no-store, то есть каждый запрос свежий.

ОТКУДА БЕРЁМ МАТЧИ. Со страницы СТРАНЫ, а не лиги:
    https://www.marathonbet.ru/su/betting/Football/UAE
Так мы не зависим от числового id турнира: LEAGUE_URL ниже содержит
"Pro+League+-+349749", и этот id живёт один сезон -- в межсезонье он
сменится, и ссылка отдаст 404 или чужой турнир. На странице страны лежат
все турниры ОАЭ сразу, и нужный отбирается по ИМЕНИ ветки в data-event-path
(сегмент "Pro League"), а имя между сезонами стабильно.

Отбор по имени ветки здесь не украшение, а защита. На той же странице висит
"Pro League Reserves", и его команды называются "Ajman Reserves",
"Al Nasr Dubai Reserves". norm_team их НЕ отсекает: у него есть запасной
проход по вхождению подстроки, и 'al nasr dubai' входит в
'al nasr dubai reserves'. Резерв уехал бы в котировки основы. Поэтому
сначала фильтр по ветке, и только потом norm_team.

ГДЕ ВЗЯТЬ id ЛИГИ ЗАНОВО, если он всё же понадобится: открыть
/su/betting/Football/UAE, взять любую строку матча, посмотреть
data-event-path -- "Football/UAE/Pro+League/<Home>+vs+<Away>+-+<treeId>";
id самой лиги виден в href на заголовке блока турнира.

ИМЕНА КОМАНД берём из data-event-path -- там АНГЛИЙСКИЕ имена
("Al+Wahda+Abu+Dhabi+vs+Al+Sharjah"). В data-event-name лежат русские
("Аль-Вахда Абу-Даби"), их norm_team тоже понимает, но английские ближе к
канону 365scores и меньше зависят от прихотей русской транслитерации.

ВРЕМЯ. В строке купона "10 сен 19:15" -- без года и в часовом поясе сайта.
Пояс читаем из initData ("tzPrefix":"GMT+3"), а не зашиваем: если Марафон
однажды отдаст линию в другом поясе, kickoff не уедет молча на три часа.
Год восстанавливаем по календарю (см. _kickoff) -- на стыке декабрь/январь
иначе матч улетит на год назад.

РЫНКИ. В строке купона видно 10 исходов, полная роспись матча (~800
исходов, ~700 КБ) лежит на странице самого матча. deep=True тянет её для
каждого матча лиги -- это 5 запросов и ~3.5 МБ на снимок.

Самое важное про фору у Марафона -- у него ТРИ разных рынка форы, и путать
их нельзя, потому что по-разному считается ничья с учётом форы:

  "Победа с учетом форы"          HB_H / HB_A          -> AH (2 исхода,
      целая линия -> возврат ставки; это азиатская механика)
  "Победа с учетом азиатской форы" HB_ASN_H / HB_ASN_A -> AH (четвертные
      линии, в тексте ячейки пара "(-0.5,-1.0)" = -0.75)
  "Победа с учетом форы (3 исхода)" HB_H / HB_D / HB_A -> EH (европейская:
      ничья с учётом форы -- отдельный исход, возврата нет)

Ключ исхода у первого и третьего рынка ОДИНАКОВЫЙ
(To_Win_Match_With_Handicap{N}), различаются они только наличием HB_D в
группе -- по нему и различаем. Проверено на живой линии Аль-Айн -- Аль-Васл:
AH1-1.5 (победа в 2 мяча, 2-исходный рынок) стоила 2.625, EH1-1 (ровно тот
же реальный исход, но на 3-исходном рынке) -- 2.59. Совпадение цен
подтверждает, что стороны и знаки разобраны правильно.

Так же устроен тотал: "Тотал голов" (2 исхода, целая линия = возврат) ->
O/U, а "Тотал голов (3 исхода)" с отдельным исходом Exactly_N -- ДРУГОЙ
рынок, его Over/Under строгие, без возврата, и канонического ключа под него
нет. Отличаем по наличию Exactly_ в группе и пропускаем.

ЧТО СЛОМАЕТСЯ ПЕРВЫМ:
  * сменится сезон -> умрёт LEAGUE_URL с id 349749 (мы им не пользуемся,
    но он в комментарии как ориентир); страница страны переживёт;
  * переименуют ветку "Pro League" -> _LEAGUE_BRANCHES ниже, добавить имя;
  * уберут data-selection-price / data-selection-key -> переписывать разбор;
  * появятся новые команды лиги -> их добавляет не адаптер, а _ALIASES
    в books/__init__.py; здесь они просто молча выпадут, а __main__
    напечатает их в разделе "не распознаны".

    PYTHONIOENCODING=utf-8 python src/books/marathon.py        # полная роспись
    PYTHONIOENCODING=utf-8 python src/books/marathon.py fast   # только купон
"""
import datetime as dt
import os
import re
import sys
import time
import urllib.parse

import requests
from bs4 import BeautifulSoup

try:
    from books import Quote, norm_team, sel_total, sel_ah, sel_eh, UA
except ImportError:  # запуск файла напрямую кладёт в sys.path src/books, а не src
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from books import Quote, norm_team, sel_total, sel_ah, sel_eh, UA

BOOK = 'marathon'
BASE = 'https://www.marathonbet.ru'
COUNTRY_URL = BASE + '/su/betting/Football/UAE'
# Прямая ссылка на лигу -- только как ориентир для человека. id 349749
# сезонный, поэтому в коде не используется, матчи ищем по имени ветки.
LEAGUE_URL = BASE + '/su/betting/Football/UAE/Pro+League+-+349749'

# Имя ветки турнира в data-event-path. Строго equals, не "начинается с":
# иначе внутрь попадут "Pro League Reserves" и "Pro League Cup".
_LEAGUE_BRANCHES = {'Pro League'}

HDRS = {
    'User-Agent': UA,
    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
    'Accept-Language': 'ru-RU,ru;q=0.9,en;q=0.8',
    'Cache-Control': 'no-cache',
    'Pragma': 'no-cache',
}

TIMEOUT = 60
TRIES = 3
PAUSE = 0.5      # пауза между страницами матчей, чтобы не долбить сайт очередью


# ---------------------------------------------------------------------------
#                                   СЕТЬ
# ---------------------------------------------------------------------------
def _get(url, session=None, tries=TRIES):
    """
    GET с ретраями. Сетевую ошибку НЕ глушим: после последней попытки
    исключение улетает наверх, сборщик отличит «контора не котирует лигу»
    от «мы сломались».
    """
    get = (session or requests).get
    for i in range(tries):
        try:
            r = get(url, headers=HDRS, timeout=TIMEOUT)
            r.raise_for_status()
            # requests при отсутствии charset в заголовке молча решает, что это
            # ISO-8859-1, и русские месяцы превращаются в мусор. Страница -- UTF-8.
            enc = r.encoding or ''
            if enc.lower() in ('', 'iso-8859-1'):
                enc = 'utf-8'
            return r.content.decode(enc, 'replace')
        except Exception:
            if i == tries - 1:
                raise
            time.sleep(2 * (i + 1))


# ---------------------------------------------------------------------------
#                              ВРЕМЯ НАЧАЛА
# ---------------------------------------------------------------------------
_MONTHS = {'янв': 1, 'фев': 2, 'мар': 3, 'апр': 4, 'мая': 5, 'май': 5,
           'июн': 6, 'июл': 7, 'авг': 8, 'сен': 9, 'окт': 10, 'ноя': 11,
           'дек': 12}
# "10 сен 19:15" либо просто "16:50" (сегодняшний матч -- день не пишут)
_RE_DATE = re.compile(r'(?:(\d{1,2})\s+([а-яё]{3})[а-яё.]*\s+)?(\d{1,2}):(\d{2})')
_RE_TZ = re.compile(r'"tzPrefix"\s*:\s*"GMT([+-]?\d+)"')


def _tz(html):
    """Часовой пояс, в котором сайт печатает время. По умолчанию московский."""
    m = _RE_TZ.search(html)
    try:
        return dt.timezone(dt.timedelta(hours=int(m.group(1))))
    except Exception:
        return dt.timezone(dt.timedelta(hours=3))


def _kickoff(text, tz):
    """'10 сен 19:15' -> unix-секунды. Год Марафон не пишет, восстанавливаем."""
    m = _RE_DATE.search((text or '').lower().replace('ё', 'е'))
    if not m:
        return None
    now = dt.datetime.now(tz)
    hh, mi = int(m.group(3)), int(m.group(4))
    if not m.group(1):
        # день не указан -- значит сегодня
        return now.replace(hour=hh, minute=mi, second=0, microsecond=0).timestamp()
    mon = _MONTHS.get(m.group(2))
    if not mon:
        return None
    day = int(m.group(1))
    for year in (now.year, now.year + 1, now.year - 1):
        try:
            d = dt.datetime(year, mon, day, hh, mi, tzinfo=tz)
        except ValueError:      # 29 февраля невисокосного года
            continue
        # линия смотрит вперёд: прошлое дальше двух месяцев -- это
        # на самом деле следующий год (случай «декабрь смотрит на январь»)
        if (d - now).days >= -60:
            return d.timestamp()
    return None


# ---------------------------------------------------------------------------
#                          РАЗБОР СТРОКИ МАТЧА
# ---------------------------------------------------------------------------
_RE_TREE_TAIL = re.compile(r'\s+-\s+\d+$')      # хвост " - 31533011" в пути


def _row_info(row):
    """
    Строка купона -> (event_id, ссылка на матч, raw_home, raw_away) или None,
    если это не матч нужной ветки.
    """
    path = row.get('data-event-path') or ''
    parts = [urllib.parse.unquote_plus(p) for p in path.split('/')]
    # ожидаем Football / UAE / <ветка> / <Home> vs <Away> - <treeId>
    if len(parts) < 4 or parts[2] not in _LEAGUE_BRANCHES:
        return None
    pair = _RE_TREE_TAIL.sub('', parts[3]).split(' vs ')
    if len(pair) != 2:
        return None
    # BeautifulSoup приводит имена атрибутов к нижнему регистру,
    # поэтому не data-event-eventId, а data-event-eventid
    ev = row.get('data-event-eventid')
    if not ev:
        return None
    # Ссылку на роспись берём готовую из строки, а не клеим из path: в path
    # пробел закодирован как '+', и любая самодельная перекодировка ломает
    # адрес (сайт отдаёт 404 на %20).
    link = row.select_one('a.member-link[href]')
    href = link['href'] if link else '/su/betting/' + path
    return ev, href, pair[0].strip(), pair[1].strip()


def _date_text(row):
    node = row.select_one('.date .date-wrapper') or row.select_one('.date')
    return node.get_text(' ', strip=True) if node else ''


# ---------------------------------------------------------------------------
#                          РАЗБОР ИСХОДОВ
# ---------------------------------------------------------------------------
# Полное совпадение группы: так отсекаются тайм-рынки
# (Total_Goals_-_1st_Half), индивидуальные тоталы (Total_Goals_(First_Team))
# и Total_Goal_Minutes, у которых префикс похож.
_RE_TOTAL = re.compile(r'^Total_Goals\d*$')
_RE_ATOTAL = re.compile(r'^Asian_Total_Goals\d*$')
_RE_HCP = re.compile(r'^To_Win_Match_With_Handicap\d*$')
_RE_AHCP = re.compile(r'^To_Win_Match_With_Asian_Handicap\d*$')

_RE_OU = re.compile(r'^(Over|Under)_([\d.]+)$')
_RE_OU_ASN = re.compile(r'^(Over|Under)_([\d.]+)_([\d.]+)$')
_RE_PAREN = re.compile(r'\(([^)]*)\)')

_1X2 = {'1': '1', 'draw': 'X', '3': '2'}
# HD = Home or Draw, HA = Home or Away, AD = Away or Draw
_DC = {'HD': '1X', 'HA': '12', 'AD': 'X2'}


def _num(text):
    """'(-1.0)'->-1.0, '(0)'->0.0, '(-0.5,-1.0)'->-0.75 (четвертная линия)."""
    vals = []
    for p in text.split(','):
        p = p.strip().replace('+', '')
        if not p:
            continue
        try:
            vals.append(float(p))
        except ValueError:
            return None
    return sum(vals) / len(vals) if vals else None


def _line(span):
    """
    Линия форы. Марафон прячет её в трёх разных местах:
      * в ячейке рядом с ценой -- <div class="coeff-value">(-1.0)</div>
        (развёрнутая роспись, 2 исхода) либо просто текстом "(-1.0)<br/>"
        (верхняя строка купона);
      * в подписи СТРОКИ таблицы -- td[data-mutable-id$=_rowLabel],
        так устроен 3-исходный рынок: линия одна на всю строку.
    """
    td = span.find_parent('td')
    if td is None:
        return None
    m = _RE_PAREN.search(td.get_text(' ', strip=True))
    if not m:
        tr = span.find_parent('tr')
        lbl = tr.select_one('td[data-mutable-id$="_rowLabel"]') if tr else None
        if lbl is None:
            return None
        m = _RE_PAREN.search(lbl.get_text(' ', strip=True))
        if not m:
            return None
    return _num(m.group(1))


def _harvest(scope, event_id):
    """
    Всё, что найдено в куске DOM, -> {канонический ключ исхода: кэф}.

    Одна и та же ставка может встретиться дважды (строка купона дублирует
    начало полной росписи) и даже прийти с двух разных рынков конторы.
    Берём МАКСИМУМ: это одна и та же контора и один и тот же исход, поставить
    можно по любой из цен, значит для нас честная цена Марафона -- лучшая.
    """
    items = []
    groups = {}
    for span in scope.select('span[data-selection-key]'):
        key = span.get('data-selection-key') or ''
        ev, _, rest = key.partition('@')
        if ev != event_id or not rest:
            continue
        grp, _, out = rest.partition('.')     # линия тотала тоже с точкой,
        if not out:                           # поэтому partition, не split
            continue
        try:
            price = float(span.get('data-selection-price'))
        except (TypeError, ValueError):
            continue
        if price <= 1.0:
            continue
        items.append((span, grp, out, price))
        groups.setdefault(grp, set()).add(out)

    res = {}

    def put(sel, price):
        if sel and (sel not in res or price > res[sel]):
            res[sel] = price

    for span, grp, out, price in items:
        if grp == 'Match_Result':
            put(_1X2.get(out), price)
        elif grp == 'Result':
            put(_DC.get(out), price)
        elif _RE_TOTAL.match(grp):
            # 3-исходный тотал (есть Exactly_N) -- другой рынок, пропускаем
            if any(o.startswith('Exactly_') for o in groups[grp]):
                continue
            m = _RE_OU.match(out)             # заодно отсекает Total_Goals.odd/even
            if m:
                put(sel_total(m.group(1) == 'Over', float(m.group(2))), price)
        elif _RE_ATOTAL.match(grp):
            m = _RE_OU_ASN.match(out)         # 'Over_2_2.5' -> линия 2.25
            if m:
                line = (float(m.group(2)) + float(m.group(3))) / 2
                put(sel_total(m.group(1) == 'Over', line), price)
        elif _RE_HCP.match(grp):
            team = {'HB_H': 1, 'HB_A': 2}.get(out)   # HB_D канонического ключа не имеет
            if not team:
                continue
            line = _line(span)
            if line is None:
                continue
            if 'HB_D' in groups[grp]:
                # 3-исходный рынок: подпись строки -- фора ПЕРВОЙ команды,
                # второй достаётся зеркальная. Дробной линии тут быть не может
                # (ничьей с учётом форы не существует) -- если появилась,
                # значит мы неверно прочли рынок, и лучше промолчать.
                if line != int(line):
                    continue
                put(sel_eh(team, line if team == 1 else -line), price)
            else:
                # 2 исхода: у каждой ячейки своя линия и уже со своим знаком
                put(sel_ah(team, line), price)
        elif _RE_AHCP.match(grp):
            team = {'HB_ASN_H': 1, 'HB_ASN_A': 2}.get(out)
            line = _line(span) if team else None
            if line is not None:
                put(sel_ah(team, line), price)
    return res


# ---------------------------------------------------------------------------
#                                  СБОР
# ---------------------------------------------------------------------------
def _collect(deep=True):
    """-> (список Quote, [нераспознанные пары имён]). Внутренняя, для __main__."""
    out, unmapped = [], []
    with requests.Session() as session:
        html = _get(COUNTRY_URL, session)
        tz = _tz(html)
        soup = BeautifulSoup(html, 'html.parser')

        for row in soup.select('div.coupon-row[data-event-path]'):
            info = _row_info(row)
            if info is None:
                continue
            event_id, href, raw_home, raw_away = info
            home, away = norm_team(raw_home), norm_team(raw_away)
            if not home or not away:
                # не ошибка: молодёжь, кубковые и просто новая команда лиги.
                # Угадывать имя нельзя -- добавлять в _ALIASES books/__init__.py.
                unmapped.append((raw_home, raw_away))
                continue

            kickoff = _kickoff(_date_text(row), tz)
            sels = _harvest(row, event_id)    # 10 исходов из строки купона

            if deep:
                time.sleep(PAUSE)
                page = _get(BASE + href, session)
                for sel, price in _harvest(BeautifulSoup(page, 'html.parser'),
                                           event_id).items():
                    if sel not in sels or price > sels[sel]:
                        sels[sel] = price

            for sel, price in sels.items():
                out.append(Quote(book=BOOK, home=home, away=away, sel=sel,
                                 price=price, kickoff=kickoff,
                                 raw_home=raw_home, raw_away=raw_away))
    return out, unmapped


def fetch(deep=True):
    """Линия Marathonbet по лиге ОАЭ. deep=False -- только строка купона."""
    return _collect(deep)[0]


# ---------------------------------------------------------------------------
if __name__ == '__main__':
    deep = 'fast' not in sys.argv[1:]
    t0 = time.time()
    quotes, unmapped = _collect(deep)

    games = {}
    for q in quotes:
        games.setdefault((q.home, q.away, q.kickoff, q.raw_home, q.raw_away),
                         {})[q.sel] = q.price

    print(f'{BOOK}: матчей {len(games)}, котировок {len(quotes)}, '
          f'{"полная роспись" if deep else "строка купона"}, {time.time() - t0:.1f} c')

    for (home, away, ko, rh, ra), sels in sorted(
            games.items(), key=lambda kv: kv[0][2] or 0):
        when = (dt.datetime.fromtimestamp(ko).strftime('%d.%m %H:%M')
                if ko else '  ??  ')
        print(f'\n{home} - {away}   {when} (местное)   [{rh} vs {ra}]')
        tot = sorted(s for s in sels if s[0] in 'OU' and s[1:2].isdigit())
        ah = sorted(s for s in sels if s.startswith('AH'))
        eh = sorted(s for s in sels if s.startswith('EH'))
        print('  исход  ' + '  '.join(f'{s}={sels[s]:g}'
                                      for s in ('1', 'X', '2') if s in sels))
        print('  дв.шанс' + '  '.join(f' {s}={sels[s]:g}'
                                      for s in ('1X', '12', 'X2') if s in sels))
        print(f'  тоталы ({len(tot)}): ' + '  '.join(f'{s}={sels[s]:g}' for s in tot))
        print(f'  форы AH ({len(ah)}): ' + '  '.join(f'{s}={sels[s]:g}' for s in ah))
        print(f'  форы EH ({len(eh)}): ' + '  '.join(f'{s}={sels[s]:g}' for s in eh))

    if unmapped:
        print('\nне распознаны norm_team (проверь _ALIASES в books/__init__.py):')
        for rh, ra in unmapped:
            print(f'  {rh} vs {ra}')
