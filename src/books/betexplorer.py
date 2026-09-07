# -*- coding: utf-8 -*-
"""
Снятие линии BetExplorer по чемпионату ОАЭ.

ЧЕМ ЭТОТ АДАПТЕР ОТЛИЧАЕТСЯ ОТ ОСТАЛЬНЫХ. BetExplorer -- не контора, а
витрина сравнения. Одна строка таблицы = одна контора, поэтому модуль
возвращает котировки НЕ под одним ключом, а под 'bx:<slug>' для каждой
конторы внутри ('bx:1xbet', 'bx:888sport', 'bx:william-hill', ...). Ровно
это и нужно консенсусу из docstring пакета: максимум цены по многим конторам
берётся не с одной мягкой линии, а с широкого фронта. Ставить у них нельзя
(в BETTABLE их нет) -- эти цены уточняют оценку истинной вероятности.

ПОЧЕМУ ИМЕННО ЭТИ ЭНДПОИНТЫ. Логина и cookies не нужно, но браузерный
User-Agent ОБЯЗАТЕЛЕН: с дефолтным UA python-requests сайт отдаёт 404.

  1) СПИСОК МАТЧЕЙ ТУРА
     GET /football/united-arab-emirates/uae-league/fixtures/
     Обычный HTML. Внимание: путь /pro-league/ НЕ существует (301 на
     главную), правильный slug лиги -- /uae-league/. Взять slug заново:
     betexplorer.com -> Football -> United Arab Emirates, там же видно
     соседние турниры "Pro League U23" и "Division 1" -- их брать НЕЛЬЗЯ,
     это молодёжка и второй дивизион.
     Из строки таблицы нужны: id матча (8 символов в href), имена команд
     (два <span> внутри a.in-match) и время в td.table-main__datetime.

  2) КОЭФФИЦИЕНТЫ ОДНОГО МАТЧА, ПО ОДНОМУ ЗАПРОСУ НА ТИП РЫНКА
     GET /match-odds/{matchId}/0/{bettype}/odds/?lang=en
     Ноль в пути -- период (0 = основное время). Ответ -- JSON вида
     {"odds": "<кусок html-таблицы>"}, то есть парсить надо HTML ВНУТРИ
     json-поля, отсюда двухступенчатый разбор ниже.
     Живьём в меню сайта шесть типов: 1x2, ou, ah, dnb, dc, bts. Берём
     четыре, которые ложатся в контракт:
        1x2 -> 1 / X / 2
        dc  -> 1X / 12 / X2
        ou  -> тоталы, по tbody на линию (видели 0.50 ... 5.50)
        ah  -> азиатские форы, по tbody на линию (видели -3.50 ... +1.50)
     dnb (draw-no-bet) и bts (обе забьют) в контракте ключей не имеют --
     пропускаем. Европейской форы (три исхода) BetExplorer не котирует
     вообще, поэтому sel_eh здесь не используется; появится -- добавлять
     надо отдельным bettype, а не пытаться вывести из ah.

ГДЕ БРАТЬ ЛИНИЮ У КОТИРОВКИ -- ТРИ ИСТОЧНИКА, ИМЕННО В ЭТОМ ПОРЯДКЕ.
Это главная тонкость файла, на ней ломаются наивные парсеры:
  а) id блока <tbody id="all-odds-{line}"> -- 'all-odds--0.75' даёт -0.75.
     Работает почти всегда, КРОМЕ одного блока: главная линия тотала
     приезжает как id="all-odds-ou" (буквы вместо числа);
  б) атрибут data-hcp="E-{bt}-{scope}-0-{line}-0" на ячейке цены. Как раз
     закрывает случай "all-odds-ou" (там data-hcp = E-2-2-0-2.5-0). Но у
     части ячеек этого атрибута нет вовсе, так что первым его ставить
     нельзя;
  в) текст td.table-main__doubleparameter. САМЫЙ НЕНАДЁЖНЫЙ: четвертные
     форы сайт печатает расщеплённой парой -- линия -2.75 показана как
     "-2.5, -3", а целые идут со знаком плюс ("+1"). Поэтому текст только
     как последний шанс, и пару приходится усреднять обратно.
Контракт требует отдавать четверти как есть (расщепление делает pricing),
так что -2.75 мы и записываем как AH1-2.75.

СТОРОНА ФОРЫ. Колонки таблицы ah подписаны "Handicap | 1 | 2", линия дана
со стороны ХОЗЯЕВ. Проверено на живой линии Al Jazira -- Al Nasr (хозяева
фавориты, 1X2 = 1.40): на линии 0 цены 1.13 / 4.65, на -0.5 -> 1.50 / 2.40,
на +0.25 -> 1.10 / 5.09. Сдвиг цен в правильную сторону, значит знак наш.
Отсюда sel_ah(1, line) и sel_ah(2, -line).

ВРЕМЯ НАЧАЛА. В HTML год не печатается ("10.09. 14:40"), и время дано в
часовом поясе сайта, а не в UTC. Пояс не зашиваем: страница отдаёт
собственное "сейчас" вызовом timezone_update('7,9,2026,5,31,29') (это
D,M,Y,H,M,S) -- сравниваем с нашим UTC и округляем разницу до часа. Сейчас
получается UTC+1; сверено с независимым источником (kickoff 1789047600 =
13:40 UTC при показанных 14:40). Год берём от этого же "сейчас" с поправкой
на переход через Новый год. Если сайт перестанет звать timezone_update --
останется дефолт UTC+1, см. _TZ_FALLBACK.

ЕЩЁ ОДНА ЛОВУШКА РАЗМЕТКИ: у матчей с одинаковым временем начала дата
печатается только в первой строке, у остальных в td.table-main__datetime
стоит &nbsp;. Пустую ячейку надо наследовать от предыдущей строки, иначе
половина тура останется без kickoff.

ЧТО СЛОМАЕТСЯ ПЕРВЫМ. По убыванию вероятности:
  - смена сезона: slug /uae-league/ обычно переживает её, но состав лиги
    меняется -- вылетевшая команда исчезает, пришедшей нет в _ALIASES, и
    её матч молча выпадет. Видно по строке "НЕ РАСПОЗНАНЫ" в выводе ниже,
    лечится синонимом в src/books/__init__.py, НЕ здесь;
  - переименование css-классов таблицы. Страховка: колонки опознаются по
    подписям заголовков (th.table-main__detail-odds), а не по номеру, так
    что перестановка колонок переживается; полная смена вёрстки -- нет;
  - появление нового типа рынка или исчезновение ah/ou у части матчей --
    это не ошибка, просто меньше котировок;
  - блокировка по User-Agent: ответ 404 на всё. Первым делом проверять UA.

    PYTHONIOENCODING=utf-8 python src/books/betexplorer.py
"""
import datetime as dt
import json
import os
import re
import sys
import time
import urllib.error
import urllib.request

try:
    from books import Quote, norm_team, sel_total, sel_ah, sel_eh, UA
except ImportError:  # прямой запуск файла: python src/books/betexplorer.py
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from books import Quote, norm_team, sel_total, sel_ah, sel_eh, UA

from bs4 import BeautifulSoup

BOOK_PREFIX = 'bx:'          # book = 'bx:<slug конторы>', см. docstring
SITE = 'https://www.betexplorer.com'
LEAGUE = '/football/united-arab-emirates/uae-league/'
FIXTURES = SITE + LEAGUE + 'fixtures/'
ODDS = SITE + '/match-odds/{mid}/0/{bt}/odds/?lang=en'

# Пояс сайта, если самокалибровка не сработает (сейчас живьём именно UTC+1).
_TZ_FALLBACK = 3600.0

# Темп: сайт терпит 1-2 запроса в секунду, ответы ou/ah тяжёлые (0.5 МБ).
PAUSE = 0.7

HDRS = {
    'User-Agent': UA,                     # без браузерного UA сайт отдаёт 404
    'Accept-Language': 'en-US,en;q=0.9',
    'Referer': SITE + '/',
}
# Ajax-эндпоинт /match-odds/ ждёт XHR-заголовок, с ним ответ стабильнее
HDRS_XHR = dict(HDRS, **{'X-Requested-With': 'XMLHttpRequest',
                         'Accept': 'application/json, text/plain, */*'})

# Сколько колонок цен ждём и как они подписаны в шапке таблицы. Подписи --
# основной способ разметки колонок; порядок ниже нужен лишь как запасной,
# если шапку не удалось прочитать.
_MARKETS = {
    '1x2': (3, {'1': '1', 'X': 'X', '2': '2'}),
    'dc':  (3, {'1X': '1X', '12': '12', 'X2': 'X2'}),
    # у тоталов и фор ключ исхода зависит от линии, поэтому здесь -- роль
    'ou':  (2, {'OVER': 'over', 'UNDER': 'under'}),
    'ah':  (2, {'1': 'home', '2': 'away'}),
}
BETTYPES = ('1x2', 'dc', 'ou', 'ah')


# ---------------------------------------------------------------------------
#                                  СЕТЬ
# ---------------------------------------------------------------------------
def _get(url, headers=HDRS, tries=3, timeout=90):
    """GET с ретраями. Сетевую ошибку НЕ глушим -- пусть летит наверх."""
    for i in range(tries):
        try:
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return r.read().decode('utf-8', 'replace')
        except Exception:
            if i == tries - 1:
                raise
            time.sleep(1.5 * (i + 1))


def _get_odds_html(mid, bt):
    """Ответ /match-odds/ -- это JSON {"odds": "<html>"}; отдаём вложенный html."""
    txt = _get(ODDS.format(mid=mid, bt=bt), headers=HDRS_XHR)
    return (json.loads(txt) or {}).get('odds') or ''


# ---------------------------------------------------------------------------
#                        ВРЕМЯ: ПОЯС САЙТА И ГОД
# ---------------------------------------------------------------------------
def _site_now(html):
    """
    'Сейчас' по часам сайта из вызова timezone_update('D,M,Y,H,M,S').
    Возвращаем (наивный datetime в поясе сайта, смещение пояса в секундах).
    Смещение считаем сами -- зашивать пояс нельзя, сайт его меняет.
    """
    m = re.search(r"timezone_update\('(\d+),(\d+),(\d+),(\d+),(\d+),(\d+)'", html)
    utc = dt.datetime.now(dt.timezone.utc).replace(tzinfo=None)
    if not m:
        return utc + dt.timedelta(seconds=_TZ_FALLBACK), _TZ_FALLBACK
    d, mo, y, hh, mi, ss = (int(x) for x in m.groups())
    try:
        site = dt.datetime(y, mo, d, hh, mi, ss)
    except ValueError:
        return utc + dt.timedelta(seconds=_TZ_FALLBACK), _TZ_FALLBACK
    off = round((site - utc).total_seconds() / 3600.0) * 3600.0
    if abs(off) > 14 * 3600:            # явная чушь -- значит, формат сменился
        return utc + dt.timedelta(seconds=_TZ_FALLBACK), _TZ_FALLBACK
    return site, off


def _kickoff(text, site_now, off):
    """'10.09. 14:40' (часы сайта, год не напечатан) -> unix-секунды UTC."""
    m = re.search(r'(\d{1,2})\.(\d{1,2})\.\s*(\d{4})?\s*(\d{1,2}):(\d{2})', text or '')
    if not m:
        return None
    d, mo, y, hh, mi = m.group(1), m.group(2), m.group(3), m.group(4), m.group(5)
    year = int(y) if y else site_now.year
    for _ in range(2):
        try:
            local = dt.datetime(year, int(mo), int(d), int(hh), int(mi))
        except ValueError:              # 29.02 не того года
            return None
        if y:
            break
        # год не напечатан: тур декабря, снятый в январе (и наоборот)
        if (local - site_now).days < -180:
            year += 1
            continue
        if (local - site_now).days > 180:
            year -= 1
            continue
        break
    return local.replace(tzinfo=dt.timezone.utc).timestamp() - off


def _created(text, off):
    """data-created='07,09,2026,05,09' (часы сайта) -> unix-секунды UTC."""
    p = (text or '').split(',')
    if len(p) < 5:
        return None
    try:
        local = dt.datetime(int(p[2]), int(p[1]), int(p[0]), int(p[3]), int(p[4]))
    except ValueError:
        return None
    return local.replace(tzinfo=dt.timezone.utc).timestamp() - off


# ---------------------------------------------------------------------------
#                            СПИСОК МАТЧЕЙ ТУРА
# ---------------------------------------------------------------------------
_HREF = re.compile(r'^' + re.escape(LEAGUE) + r'[a-z0-9-]+/([A-Za-z0-9]{8})/$')


def _fixtures(html):
    """
    -> (список матчей, список нераспознанных пар имён).
    Матч отбрасывается молча, если norm_team не знает хотя бы одну команду:
    в этом же разделе сайта попадаются кубковые и молодёжные составы.
    """
    soup = BeautifulSoup(html, 'html.parser')
    site_now, off = _site_now(html)
    out, unknown, seen = [], [], set()
    last_dt = ''
    for tr in soup.find_all('tr'):
        cell = tr.find('td', class_='table-main__datetime')
        a = tr.find('a', class_='in-match')
        if not (cell and a):
            continue
        m = _HREF.match(a.get('href') or '')
        if not m or m.group(1) in seen:
            continue
        # дата печатается только у первого матча каждого времени начала,
        # у соседей в ячейке стоит &nbsp; -- наследуем предыдущее значение
        txt = cell.get_text(' ', strip=True).replace('\xa0', ' ').strip()
        if txt:
            last_dt = txt
        spans = [s.get_text(strip=True) for s in a.find_all('span')]
        if len(spans) < 2:              # запасной разбор "Home - Away"
            spans = [p.strip() for p in a.get_text(' ', strip=True).split(' - ', 1)]
        if len(spans) < 2:
            continue
        raw_h, raw_a = spans[0], spans[1]
        home, away = norm_team(raw_h), norm_team(raw_a)
        if not (home and away):
            unknown.append(f'{raw_h} - {raw_a}')
            continue
        seen.add(m.group(1))
        out.append(dict(mid=m.group(1), home=home, away=away,
                        raw_home=raw_h, raw_away=raw_a,
                        kickoff=_kickoff(last_dt, site_now, off)))
    return out, unknown, off


# ---------------------------------------------------------------------------
#                       РАЗБОР ТАБЛИЦЫ КОЭФФИЦИЕНТОВ
# ---------------------------------------------------------------------------
def _book_slug(name):
    """'William Hill' / 'Stake.com' -> 'william-hill' / 'stake-com'."""
    out = []
    for ch in (name or '').strip().lower():
        if ch.isalnum():
            out.append(ch)
        elif out and out[-1] != '-':
            out.append('-')
    return ''.join(out).strip('-')


def _columns(soup, bt):
    """
    Колонки цен -> роли, по подписям шапки (th.table-main__detail-odds).
    У ou/ah шапка повторяется несколько раз, берём первую группу.
    Если подписи не читаются -- документированный порядок из _MARKETS.
    """
    n, labels = _MARKETS[bt]
    heads = [th.get_text(strip=True).upper()
             for th in soup.find_all('th', class_='table-main__detail-odds')][:n]
    roles = [labels.get(h) for h in heads]
    if len(roles) == n and all(roles):
        return roles
    return list(labels.values())        # порядок словаря = порядок колонок


def _line(tbody, cell, param_td):
    """
    Линия котировки. Три источника по убыванию надёжности, см. docstring:
    id блока -> data-hcp -> текст параметра (там четверти расщеплены).
    """
    tid = (tbody.get('id') or '') if tbody is not None else ''
    if tid.startswith('all-odds-'):
        try:
            return float(tid[len('all-odds-'):])
        except ValueError:
            pass                        # 'all-odds-ou' -- главная линия тотала
    m = re.match(r'^E-\d+-\d+-\d+-(-?\d+(?:\.\d+)?)-\d+$', cell.get('data-hcp') or '')
    if m:
        return float(m.group(1))
    txt = (param_td.get_text(strip=True) if param_td is not None else '')
    txt = txt.replace('+', '').strip()
    if not txt:
        return None
    try:                                # "-2.5, -3" -- расщеплённая четверть
        parts = [float(p) for p in txt.split(',') if p.strip()]
    except ValueError:
        return None
    return sum(parts) / len(parts) if parts else None


def _sel(bt, role, line):
    """Роль колонки + линия -> канонический ключ исхода."""
    if bt in ('1x2', 'dc'):
        return role
    if line is None:
        return None
    if bt == 'ou':
        if line <= 0:                   # тотал 0 не бывает, это мусор разметки
            return None
        return sel_total(role == 'over', line)
    # ah: линия напечатана со стороны хозяев, гостям меняем знак
    return sel_ah(1, line) if role == 'home' else sel_ah(2, -line)


def _parse_odds(html, bt, match, off):
    """Таблица одного типа рынка одного матча -> список Quote."""
    soup = BeautifulSoup(html, 'html.parser')
    roles = _columns(soup, bt)
    out = []
    for tr in soup.find_all('tr', attrs={'data-bid': True}):
        cells = [td for td in tr.find_all('td')
                 if 'table-main__detail-odds' in (td.get('class') or [])
                 and td.get('data-odd')]
        if len(cells) != len(roles):
            continue                    # строка-заглушка либо чужая разметка
        span = tr.find('span', class_='in-bookmaker-logo')
        title = (span.get('title') if span else None) or ''
        slug = _book_slug(cells[0].get('data-bookie') or title)
        if not slug:
            continue
        tbody = tr.find_parent('tbody')
        param_td = tr.find('td', class_='table-main__doubleparameter')
        for role, td in zip(roles, cells):
            try:
                price = float(td.get('data-odd'))
            except (TypeError, ValueError):
                continue
            if price <= 1.0:            # цена ниже 1.0 -- битая ячейка
                continue
            sel = _sel(bt, role, _line(tbody, td, param_td))
            if not sel:
                continue
            out.append(Quote(
                book=BOOK_PREFIX + slug,
                home=match['home'], away=match['away'], sel=sel, price=price,
                kickoff=match['kickoff'],
                raw_home=match['raw_home'], raw_away=match['raw_away'],
                extra=dict(match_id=match['mid'], bettype=bt,
                           bookmaker=title or slug,
                           # когда контора последний раз двигала эту цену --
                           # тот же сигнал, что md у БЕТСИТИ
                           created=_created(td.get('data-created'), off)),
            ))
    return out


# ---------------------------------------------------------------------------
#                              ТОЧКА ВХОДА
# ---------------------------------------------------------------------------
def fetch():
    """Вся линия лиги ОАЭ с BetExplorer -> list[Quote]."""
    matches, _unknown, off = _fixtures(_get(FIXTURES))
    out = []
    for match in matches:
        for bt in BETTYPES:
            time.sleep(PAUSE)
            html = _get_odds_html(match['mid'], bt)
            got = _parse_odds(html, bt, match, off) if html else []
            out.extend(got)
            # Матч в календаре есть, а котировок нет ни у кого -- сайт прямо
            # пишет "there isn't any bookmaker offering odds for this match"
            # и отдаёт пустую таблицу на ЛЮБОЙ bettype (в туре 10-13.09.2026
            # так висели United FC -- Shabab Al-Ahli и Hatta -- Al Dhafra).
            # 1x2 идёт первым и есть всегда, когда есть хоть что-то, поэтому
            # пустой 1x2 -- достаточное основание не дёргать тяжёлые ou/ah.
            if bt == BETTYPES[0] and not got:
                break
    return out


if __name__ == '__main__':
    t0 = time.time()
    page = _get(FIXTURES)
    matches, unknown, off = _fixtures(page)
    print(f'BetExplorer: {FIXTURES}')
    print(f'пояс сайта UTC{off / 3600:+.0f}, матчей в туре {len(matches)}')
    if unknown:
        # дыры в словаре синонимов: матч тихо выпадает из линии
        print('НЕ РАСПОЗНАНЫ (нет в _ALIASES, матч пропущен): '
              + '; '.join(unknown))

    quotes = fetch()

    def _fam(s):
        if s in ('1', 'X', '2'):
            return '1X2'
        if s in ('1X', '12', 'X2'):
            return 'ДШ'
        return 'тотал' if s[0] in 'OU' else 'фора'

    books = {}
    for q in quotes:
        books[q.book] = books.get(q.book, 0) + 1
    print(f'котировок {len(quotes)}, контор внутри {len(books)}')
    print('  ' + '  '.join(f'{b}({n})' for b, n in
                           sorted(books.items(), key=lambda kv: -kv[1])))

    games = {}
    for q in quotes:
        games.setdefault((q.home, q.away), []).append(q)
    # матч в календаре есть, а линии нет ни у одной конторы -- это норма
    # между турами, а не поломка парсера
    dry = [m for m in matches if (m['home'], m['away']) not in games]
    if dry:
        print('БЕЗ ЛИНИИ (никто ещё не котирует): '
              + '; '.join(f"{m['raw_home']} — {m['raw_away']}" for m in dry))

    for (home, away), qs in sorted(games.items(),
                                   key=lambda kv: kv[1][0].kickoff or 0):
        ko = qs[0].kickoff
        when = (dt.datetime.fromtimestamp(ko, dt.timezone.utc)
                  .strftime('%Y-%m-%d %H:%M UTC') if ko else '?')
        by = {}
        for q in qs:
            by.setdefault(_fam(q.sel), []).append(q)
        counts = ', '.join(f'{k} {len(v)}' for k, v in sorted(by.items()))
        tot = sorted({q.sel[1:] for q in qs if q.sel[0] in 'OU'}, key=float)
        ah = sorted({q.sel[3:] for q in qs if q.sel.startswith('AH1')}, key=float)
        print(f'\n{home} — {away}   {when}   [{qs[0].raw_home} — {qs[0].raw_away}]')
        print(f'  котировок {len(qs)}: {counts}')
        print(f'  линий тоталов {len(tot)} ({", ".join(tot[:9])}...)')
        print(f'  линий фор {len(ah)} ({", ".join(ah[:9])}...)')
        # лучшая цена по каждому ключевому исходу и кто её даёт
        best = {}
        for q in qs:
            if q.price > best.get(q.sel, (0, ''))[0]:
                best[q.sel] = (q.price, q.book)
        show = ['1', 'X', '2', '1X', '12', 'X2', 'O2.5', 'U2.5',
                'AH1-0.5', 'AH2+0.5', 'AH1-1', 'AH2+1']
        for s in show:
            if s in best:
                print(f'    {s:<9} {best[s][0]:>7.2f}  максимум у {best[s][1]}')
    print(f'\nвсего {time.time() - t0:.0f} с')
