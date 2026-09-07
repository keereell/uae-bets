# -*- coding: utf-8 -*-
"""
Снятие линии ОАЭ с агрегатора Oddspedia -- сразу два десятка контор за раз.

ЗАЧЕМ ОН НУЖЕН. Остальные адаптеры проекта -- российские конторы, а прибыль
в литературе берут максимумом цены по ШИРОКОМУ рынку (Kaunitz 2017 -- 32
конторы). Oddspedia отдаёт по матчу линии ~20-25 европейских и азиатских
букмекеров, которых у нас иначе нет вообще. Ставить у них нельзя (в BETTABLE
их нет), но для оценки истинной вероятности они бесценны: консенсус по 25
конторам заметно острее, чем по 8 российским.

Ключ каждой цены пишется как 'op:<bookie_slug>' -- префикс, чтобы никогда не
перепутать «цену, которую мы видим у агрегатора» с «конторой, где мы играем».

------------------------------------------------------------------------------
ШАГ 1. ID МАТЧЕЙ -- ТОЛЬКО ИЗ HTML, ПРОГРАММНОГО СПИСКА НЕТ
------------------------------------------------------------------------------
    GET https://oddspedia.com/football/united-arab-emirates/pro-league  (~690 КБ)

Внутренний эндпоинт списка /api/v1/getMatchList проверен живьём и отдаёт 400
на любой набор параметров, скопированных из их же бандла (app.store.js,
геттер matchListParametersForLeaguePredictions: geoCode, sport, category,
league, seasonId, status, startDate, endDate, page, perPage). Похоже, фронт
подписывает запрос чем-то, чего в query нет. Поэтому id матчей берём из
серверного payload'а Nuxt, вшитого в HTML:

    window.__NUXT__=(function(a,b,c,...){ ...; return {...} }("Al Ain",1109,...));

Это минифицированный Nuxt 2: все повторяющиеся литералы вынесены в аргументы
функции, а в теле стоят однобуквенные ссылки (league_id:v, ht:bB). Готового
JSON нет, поэтому ниже лежит маленький парсер подмножества JS (_P): объекты,
массивы, строки, числа, true/false/null, void 0, Array(n), new Date(...) и
идентификаторы-параметры, которые потом подставляются из списка аргументов.
Нужны поля матча: id, md ("2026-09-10 16:15:00+00", UTC), ht, at.

ГДЕ ВЗЯТЬ id ЛИГИ ЗАНОВО. В разобранном payload'е: data[i].league.id рядом с
data[i].matchList. Открыть страницу лиги, найти 'league_id:' -- имя параметра
справа от двоеточия и есть ссылка на число. Сейчас это 1109. Матчи мы ищем
СНАЧАЛА по блокам data[] с matchList (не завязываясь на число), и только
если там пусто -- по всему дереву объектов с league_id == LEAGUE_ID.

------------------------------------------------------------------------------
ШАГ 2. КЭФЫ -- ЧЕТЫРЕ ЗАПРОСА НА МАТЧ
------------------------------------------------------------------------------
    GET /api/v1/getMatchOdds?geoCode=&matchId=<ID>&wettsteuer=0&geoState=
        &bookmakerGeoCode=&bookmakerGeoState=&language=en[&oddGroupId=<N>]

Один запрос -- одна группа рынков; в ответе всегда перечислены ВСЕ группы, но
котировки заполнены только у запрошенной. Берём четыре:

    без oddGroupId -> market id 1  'Full Time Result', ot_id 100,
                      odds[] = {bookie_slug, bid, o1,o2,o3} = 1 / X / 2
    oddGroupId=7   -> market id 7  'Double Chance',    ot_id 700,
                      odds[] = {o1,o2,o3} = 1X / 12 / X2
    oddGroupId=4   -> market id 4  'Total Goals',      ot_id 401,
                      odds.main[] + odds.alternative[]:
                      {name_en:"2.75", odds:{"<bid>":{bookie_slug,o1,o2}}},
                      o1 = Over, o2 = Under
    oddGroupId=3   -> market id 3  'Asian Handicap',   ot_id 301, та же форма,
                      name_en = линия СО СТОРОНЫ ХОЗЯИНА ("-0.25", "+1"),
                      o1 = хозяева на эту линию, o2 = гости на минус эту

ot_id -- период: 100/700/401/301 это матч целиком, 101/102, 701/702, 402/403,
302 -- таймы. Тайм-рынки в общий контракт не укладываются и отбрасываются.
Четвертные линии (0.25/0.75) отдаём как есть -- расщепление делает pricing.
Европейской (трёхисходной) форы Oddspedia по этой лиге не котирует, поэтому
sel_eh не используется; Draw No Bet (oddGroupId=5) -- это та же AH 0, пятый
запрос на матч ради дубля не делаем.

geoCode ПУСТОЙ обязателен: он же с DE даёт 2 конторы, с GB -- 8, с RU/UA
вообще 404. Пустой -- максимальный список.

------------------------------------------------------------------------------
CLOUDFLARE, БЕЗ ЧЕГО НЕ РАБОТАЕТ
------------------------------------------------------------------------------
Перед сайтом стоит managed challenge, и он придирчив втройне:
  1) браузерный User-Agent и Referer: https://oddspedia.com/ -- иначе 403
     "Just a moment...";
  2) заголовок Accept обязателен (без него тоже 403);
  3) КУКИ. Первый запрос почти всегда 403, но в ответе приезжает __cf_bm; со
     вторым запросом он уходит обратно, и дальше всё стабильно 200. Поэтому
     тут именно urllib с общим CookieJar на весь проход, а не requests:
     requests в этих же условиях получал 403 на всех попытках подряд (шесть
     из шести), urllib с куками -- 200 пять раз из шести. Разница, судя по
     всему, в TLS-отпечатке. НЕ переписывать на requests «для единообразия».

------------------------------------------------------------------------------
КЛОНЫ
------------------------------------------------------------------------------
Часть брендов торгует одной и той же линией (проверено: у fezbet, powbet,
tooniebet и campobet цены совпадают до знака). В консенсусе это одна цена,
посчитанная четыре раза, то есть искусственный вес. Схлопываем в один book --
первый слаг по алфавиту, цена = максимум по группе (ровно то, что и так
сделает поиск лучшей цены сверху).

------------------------------------------------------------------------------
ЧТО СЛОМАЕТСЯ ПЕРВЫМ, по убыванию вероятности
------------------------------------------------------------------------------
  - Cloudflare ужесточит проверку -- станет 403 на всех попытках. Симптом:
    исключение с "403" из _get на первом же шаге. Лечится только сменой
    транспорта (куки от живого браузера), не заголовками.
  - Смена минификатора Nuxt (Nuxt 3 отдаёт payload другим форматом, без
    (function(...){...})). Симптом: ValueError 'это не (function(...)) payload'.
    Тогда парсер переписывается целиком, ищите JSON в <script type="application/json">.
  - Новый сезон -> новый league_id. Не страшно: матчи берутся из блоков
    data[].matchList страницы лиги, число 1109 -- только запасной путь.
  - Новая команда в лиге, которой нет в _ALIASES: norm_team вернёт None и
    матч молча выпадет. Видно по строке "не распознаны" в выводе ниже --
    лечится синонимом в src/books/__init__.py, НЕ здесь.
  - 429 при обходе матчей -- есть пауза и ретраи; Varnish кэширует ответы,
    так что generated_at может отставать на минуту-две (кладём его в
    Quote.extra['gen'], чтобы это было видно в снимке).

    PYTHONIOENCODING=utf-8 python src/books/oddspedia.py
"""
import calendar
import datetime as dt
import gzip
import http.cookiejar
import json
import os
import re
import sys
import time
import urllib.error
import urllib.request

try:
    from books import Quote, norm_team, sel_total, sel_ah, sel_eh, UA
except ImportError:  # прямой запуск файла: python src/books/oddspedia.py
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from books import Quote, norm_team, sel_total, sel_ah, sel_eh, UA

PREFIX = 'op:'                       # 'op:1xbet' -- цена агрегатора, не наша контора
LEAGUE_URL = 'https://oddspedia.com/football/united-arab-emirates/pro-league'
LEAGUE_ID = 1109                     # запасной опознавательный знак, см. docstring
LEAGUE_SLUGS = ('pro-league', 'uae-pro-league', 'arabian-gulf-league')
ODDS_URL = ('https://oddspedia.com/api/v1/getMatchOdds?geoCode=&matchId={mid}'
            '&wettsteuer=0&geoState=&bookmakerGeoCode=&bookmakerGeoState='
            '&language=en{grp}')

# Accept и Referer -- не косметика, без любого из них Cloudflare отдаёт 403.
HDRS = {
    'User-Agent': UA,
    'Referer': 'https://oddspedia.com/',
    'Accept': '*/*',
    'Accept-Language': 'en-US,en;q=0.9',
    'Accept-Encoding': 'gzip, deflate',
}

PAUSE = 0.4                          # между запросами к API, против 429

# группа рынка -> (oddGroupId, id рынка в ответе, ot_id полного матча)
GROUPS = (
    ('1x2',   None, 1, 100),
    ('dc',    7,    7, 700),
    ('total', 4,    4, 401),
    ('ah',    3,    3, 301),
)
_1X2_SEL = ('1', 'X', '2')           # порядок oddsnames ['Home','Draw','Away']
_DC_SEL = ('1X', '12', 'X2')         # порядок oddsnames ['1X','12','X2']

# Одна линия под разными брендами. Первые четыре группы -- из описания
# источника; пятая (20bet/ivibet) добавлена по живой выборке: цены совпадали
# во всех рынках всех матчей. Если группа окажется лишней -- просто убрать
# строку, ничего больше править не надо.
_CLONES = (
    ('1xbet', '22bet', 'megapari'),
    ('goldenbet', 'freshbet', 'mystake', 'jackbit'),
    ('fezbet', 'powbet', 'tooniebet', 'campobet'),
    ('bc-game', 'nine-casino', '4rabet'),
    ('coral', 'ladbrokes', 'bwin', 'sportingbet'),
    ('20bet', 'ivibet'),
)
_CANON = {s: min(g) for g in _CLONES for s in g}


# ---------------------------------------------------------------------------
#                      ТРАНСПОРТ (куки обязательны, см. docstring)
# ---------------------------------------------------------------------------
def _opener():
    """Свежий opener с собственной банкой кук -- никакого глобального состояния."""
    return urllib.request.build_opener(
        urllib.request.HTTPCookieProcessor(http.cookiejar.CookieJar()))


def _get(op, url, tries=4):
    """
    GET с ретраями. 403 (Cloudflare, пока не выдана __cf_bm) и 429 -- ждём и
    повторяем; всё остальное (404, 500, обрыв связи) после последней попытки
    улетает наверх исключением, как требует контракт.
    """
    for i in range(tries):
        try:
            with op.open(urllib.request.Request(url, headers=HDRS), timeout=60) as r:
                body = r.read()
                if r.headers.get('Content-Encoding') == 'gzip':
                    body = gzip.decompress(body)
                return body
        except urllib.error.HTTPError as e:
            if e.code not in (403, 429, 502, 503) or i == tries - 1:
                raise
        except Exception:
            if i == tries - 1:
                raise
        time.sleep(1.5 * (i + 1))


def _json(op, url, tries=4):
    return json.loads(_get(op, url, tries).decode('utf-8'))


# ---------------------------------------------------------------------------
#          ПАРСЕР window.__NUXT__ -- минимальное подмножество JS-литералов
# ---------------------------------------------------------------------------
class _Ref(object):
    """Ссылка на параметр функции-обёртки; подставляется вторым проходом."""
    __slots__ = ('name',)

    def __init__(self, name):
        self.name = name


_NUM = re.compile(r'-?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?')
_IDENT = re.compile(r'[A-Za-z_$][A-Za-z0-9_$]*')


class _P(object):
    def __init__(self, s, i=0):
        self.s, self.i = s, i

    def _ws(self):
        while self.i < len(self.s) and self.s[self.i] in ' \t\r\n':
            self.i += 1

    def _at(self, ch):
        self._ws()
        return self.i < len(self.s) and self.s[self.i] == ch

    def value(self):
        self._ws()
        s, i = self.s, self.i
        c = s[i]
        if c == '{':
            return self.obj()
        if c == '[':
            return self.arr()
        if c in '"\'':
            return self.str()
        if c == '-' or c.isdigit() or (c == '.' and s[i + 1].isdigit()):
            m = _NUM.match(s, i)
            self.i = m.end()
            t = m.group(0)
            return float(t) if ('.' in t or 'e' in t or 'E' in t) else int(t)
        m = _IDENT.match(s, i)
        if not m:
            raise ValueError('не значение на %d: %r' % (i, s[i:i + 40]))
        w = m.group(0)
        self.i = m.end()
        if w == 'true':
            return True
        if w == 'false':
            return False
        if w == 'null':
            return None
        if w == 'void':                       # void 0
            self._ws()
            m2 = _NUM.match(s, self.i)
            if m2:
                self.i = m2.end()
            return None
        if w == 'new':                        # new Date("2026-09-10")
            self._ws()
            m2 = _IDENT.match(s, self.i)
            self.i = m2.end()
            a = self.args() if self._at('(') else []
            return a[0] if a else None
        if self._at('('):                     # Array(5) и прочие вызовы
            a = self.args()
            if w == 'Array':
                return [None] * (a[0] if a and isinstance(a[0], int) else 0)
            return a[0] if a else None
        return _Ref(w)

    def args(self):
        """Список значений в скобках: (v, v, ...) -> [v, v, ...]"""
        self._ws()
        self.i += 1                           # '('
        out = []
        while True:
            self._ws()
            if self.s[self.i] == ')':
                self.i += 1
                return out
            out.append(self.value())
            self._ws()
            if self.s[self.i] == ',':
                self.i += 1

    def str(self):
        s, i = self.s, self.i
        q, j = s[i], i + 1
        while s[j] != q:
            j += 2 if s[j] == '\\' else 1
        raw, self.i = s[i:j + 1], j + 1
        if q == "'":                          # одинарные кавычки -> JSON-совместимые
            raw = '"' + raw[1:-1].replace('"', '\\"') + '"'
        return json.loads(raw)                # / и прочие escape'ы -- бесплатно

    def arr(self):
        self.i += 1
        out = []
        while True:
            self._ws()
            c = self.s[self.i]
            if c == ']':
                self.i += 1
                return out
            if c == ',':                      # дыра в массиве: [,,]
                self.i += 1
                continue
            out.append(self.value())
            self._ws()
            if self.s[self.i] == ',':
                self.i += 1

    def obj(self):
        self.i += 1
        out = {}
        while True:
            self._ws()
            c = self.s[self.i]
            if c == '}':
                self.i += 1
                return out
            if c == ',':
                self.i += 1
                continue
            if c in '"\'':
                k = self.str()
            else:
                m = _IDENT.match(self.s, self.i) or _NUM.match(self.s, self.i)
                k, self.i = m.group(0), m.end()
            self._ws()
            if self.s[self.i] != ':':
                raise ValueError('ожидалось ":" на %d' % self.i)
            self.i += 1
            out[k] = self.value()


def _return_pos(src, start):
    """Позиция значения при первом операторе return тела функции (строки пропускаем)."""
    i, n = start, len(src)
    while i < n:
        c = src[i]
        if c in '"\'':
            i += 1
            while src[i] != c:
                i += 2 if src[i] == '\\' else 1
            i += 1
            continue
        if c == 'r' and src.startswith('return', i) and not (
                src[i - 1].isalnum() or src[i - 1] in '_$'):
            i += 6
            while src[i] in ' \t\r\n':
                i += 1
            return i
        i += 1
    raise ValueError('в payload нет return')


def parse_nuxt(src):
    """Текст после 'window.__NUXT__=' и до '</script>' -> обычный dict."""
    src = src.strip().rstrip(';')
    m = re.match(r'\(\s*function\s*\(([^)]*)\)\s*\{', src)
    if not m:
        raise ValueError('это не (function(...){...}) payload -- сменился Nuxt')
    params = [p.strip() for p in m.group(1).split(',') if p.strip()]
    p = _P(src, _return_pos(src, m.end()))
    root = p.value()
    p._ws()
    if p.s[p.i] != '}':                       # конец тела функции, дальше аргументы
        raise ValueError('после return-объекта нет "}"')
    p.i += 1
    env = dict(zip(params, p.args()))

    done = set()

    def sub(v):
        if isinstance(v, _Ref):
            r = env.get(v.name)
            return sub(r) if isinstance(r, _Ref) else r
        if isinstance(v, dict):
            if id(v) not in done:
                done.add(id(v))
                for k in list(v):
                    v[k] = sub(v[k])
            return v
        if isinstance(v, list):
            if id(v) not in done:
                done.add(id(v))
                for i, x in enumerate(v):
                    v[i] = sub(x)
            return v
        return v

    return sub(root)


def _walk(node, key):
    """Все словари в дереве, где есть ключ key."""
    if isinstance(node, dict):
        if key in node:
            yield node
        for v in node.values():
            for x in _walk(v, key):
                yield x
    elif isinstance(node, list):
        for v in node:
            for x in _walk(v, key):
                yield x


# ---------------------------------------------------------------------------
#                        ШАГ 1: СПИСОК МАТЧЕЙ ИЗ HTML
# ---------------------------------------------------------------------------
def _kickoff(md):
    """'2026-09-10 16:15:00+00' (UTC) -> unix-секунды. Смещение всегда +00."""
    if not md or not isinstance(md, str):
        return None
    try:
        t = dt.datetime.strptime(md[:19], '%Y-%m-%d %H:%M:%S')
    except ValueError:
        return None
    return float(calendar.timegm(t.timetuple()))


def match_list(op):
    """
    -> [{'id', 'ht', 'at', 'kickoff'}] по ближайшему туру лиги ОАЭ.

    Основной путь -- блоки data[] с matchList и нужным слагом лиги; запасной --
    любые объекты дерева с league_id == LEAGUE_ID (на случай, если страница
    переедет на другую раскладку).
    """
    html = _get(op, LEAGUE_URL).decode('utf-8', 'replace')
    i = html.find('window.__NUXT__=')
    j = html.find('</script>', i)
    if i < 0 or j < 0:
        # сюда же попадём, если Cloudflare отдал 200 со страницей-заглушкой
        raise ValueError('на странице лиги нет window.__NUXT__ -- сменилась вёрстка'
                         ' или пришла заглушка (%d КБ)' % (len(html) // 1024))
    root = parse_nuxt(html[i + len('window.__NUXT__='):j])

    raw = []
    for blk in _walk(root, 'matchList'):
        lg = blk.get('league') or {}
        slug = lg.get('slug_en') or lg.get('slug')
        if slug in LEAGUE_SLUGS or lg.get('id') == LEAGUE_ID:
            raw.extend(blk.get('matchList') or [])
    if not raw:                               # запасной путь
        raw = [m for m in _walk(root, 'league_id') if m.get('league_id') == LEAGUE_ID]

    now, seen, out = time.time(), set(), []
    for m in raw:
        mid = m.get('id')
        if not isinstance(mid, int) or mid in seen:
            continue
        # уже начавшиеся и сыгранные не берём: там линия либо лайвовая, либо
        # расчётная, в консенсус прематча ей нельзя
        if m.get('inplay') or m.get('hscore') is not None:
            continue
        ko = _kickoff(m.get('md'))
        if ko is None or ko < now - 900:
            continue
        seen.add(mid)
        out.append(dict(id=mid, ht=m.get('ht') or '', at=m.get('at') or '', kickoff=ko))
    return sorted(out, key=lambda r: r['kickoff'])


# ---------------------------------------------------------------------------
#                       ШАГ 2: КЭФЫ ОДНОГО МАТЧА
# ---------------------------------------------------------------------------
def _price(v):
    """Кэфы приходят строками ('1.6289308'); мусор и пустоту отбрасываем."""
    try:
        p = float(v)
    except (TypeError, ValueError):
        return None
    return p if 1.0 < p < 1000.0 else None


def _line(v):
    """name_en тотала/форы: '2.75', '-0.25', '+1', '0'."""
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _market(payload, market_id, ot_id):
    """-> объект периода нужного рынка полного матча, либо None."""
    for mk in (payload.get('data') or {}).get('prematch') or []:
        if mk.get('id') != market_id:
            continue
        for per in mk.get('periods') or []:
            if per.get('ot_id') == ot_id:
                return per
    return None


def _flat_lines(period):
    """odds у тоталов/фор -- {'main': [...], 'alternative': [...]}, склеиваем."""
    o = period.get('odds')
    if isinstance(o, dict):
        return (o.get('main') or []) + (o.get('alternative') or [])
    return o or []


def match_odds(op, mid):
    """
    -> ([(bookie_slug, sel, price)], generated_at)

    Четыре запроса на матч: 1X2, двойной шанс, тоталы, азиатская фора.

    404 -- это НЕ поломка: так отвечает getMatchOdds, когда матч ещё не открыт
    ни у одной конторы (проверено живьём: на весь тур такими были два матча
    подряд, все четыре группы). Считаем «нет линии» и идём дальше.
    """
    got, gen = [], None
    for tag, gid, market_id, ot_id in GROUPS:
        url = ODDS_URL.format(mid=mid, grp='' if gid is None else '&oddGroupId=%d' % gid)
        try:
            j = _json(op, url)
        except urllib.error.HTTPError as e:
            if e.code == 404:
                continue
            raise
        gen = j.get('generated_at') or gen
        per = _market(j, market_id, ot_id)
        time.sleep(PAUSE)
        if not per:
            continue

        if tag in ('1x2', 'dc'):
            names = _1X2_SEL if tag == '1x2' else _DC_SEL
            for row in per.get('odds') or []:
                slug = row.get('bookie_slug')
                if not slug:
                    continue
                for n, sel in enumerate(names, 1):
                    p = _price(row.get('o%d' % n))
                    if p:
                        got.append((slug, sel, p))
            continue

        # тоталы и форы: линия в name_en, конторы -- в словаре по bid
        for ent in _flat_lines(per):
            ln = _line(ent.get('name_en'))
            if ln is None:
                continue
            for row in (ent.get('odds') or {}).values():
                slug = row.get('bookie_slug')
                if not slug:
                    continue
                p1, p2 = _price(row.get('o1')), _price(row.get('o2'))
                if tag == 'total':
                    if p1:
                        got.append((slug, sel_total(True, ln), p1))
                    if p2:
                        got.append((slug, sel_total(False, ln), p2))
                else:                          # ah: name_en -- линия хозяина
                    if p1:
                        got.append((slug, sel_ah(1, ln), p1))
                    if p2:
                        got.append((slug, sel_ah(2, -ln), p2))
    return got, gen


# ---------------------------------------------------------------------------
#                                  FETCH
# ---------------------------------------------------------------------------
def fetch():
    """Линия лиги ОАЭ у всех контор, которые видит Oddspedia. -> list[Quote]"""
    op = _opener()
    games = match_list(op)                    # сетевая ошибка тут летит наверх сразу

    out, fails = [], []
    for g in games:
        home, away = norm_team(g['ht']), norm_team(g['at'])
        if not home or not away:
            continue                          # не наша команда -- молча мимо
        try:
            rows, gen = match_odds(op, g['id'])
        except Exception as e:
            fails.append(e)                   # один матч мог отвалиться по 429
            continue
        # схлопывание клонов: одна линия под несколькими брендами = одна цена
        best = {}
        for slug, sel, price in rows:
            k = (_CANON.get(slug, slug), sel)
            if price > best.get(k, 0.0):
                best[k] = price
        for (book, sel), price in best.items():
            out.append(Quote(book=PREFIX + book, home=home, away=away, sel=sel,
                             price=price, kickoff=g['kickoff'],
                             raw_home=g['ht'], raw_away=g['at'],
                             extra={'gen': gen, 'mid': g['id']}))
    # молчать про поломку нельзя: если не вышло НИЧЕГО и были ошибки -- наверх
    if fails and not out:
        raise fails[0]
    return out


def _utc(ts):
    return dt.datetime.fromtimestamp(ts, dt.timezone.utc).strftime('%d.%m %H:%M')


if __name__ == '__main__':
    t0 = time.time()
    qs = fetch()
    games = {}
    for q in qs:
        games.setdefault((q.home, q.away), []).append(q)

    print('котировок %d, матчей %d, контор %d, за %.1f с'
          % (len(qs), len(games), len({q.book for q in qs}), time.time() - t0))

    fams = {'1X2': ('1', 'X', '2'), 'ДШ': ('1X', '12', 'X2')}
    for (h, a), qq in sorted(games.items(), key=lambda kv: kv[1][0].kickoff or 0):
        ko = _utc(qq[0].kickoff)
        sels = {q.sel for q in qq}
        tot = sorted({s[1:] for s in sels if s[0] in 'OU'}, key=float)
        ah = sorted({s[3:] for s in sels if s.startswith('AH1')}, key=float)
        print('\n%-16s - %-16s  %s UTC   контор %d, исходов %d'
              % (h, a, ko, len({q.book for q in qq}), len(sels)))
        print('   %s | %s | тоталов %d (%s) | фор %d (%s)'
              % (qq[0].raw_home, qq[0].raw_away, len(tot), ','.join(tot),
                 len(ah), ','.join(ah)))
        for fam, keys in fams.items():
            line = []
            for k in keys:
                pp = [q for q in qq if q.sel == k]
                if pp:
                    b = max(pp, key=lambda q: q.price)
                    line.append('%s %.2f (%s)' % (k, b.price, b.book[3:]))
            if line:
                print('   макс %-4s %s' % (fam, '   '.join(line)))
        for k in ('O2.5', 'U2.5', 'AH1-0.5', 'AH2+0.5'):
            pp = [q for q in qq if q.sel == k]
            if pp:
                b = max(pp, key=lambda q: q.price)
                print('   макс %-8s %.2f (%s)  [%d контор]'
                      % (k, b.price, b.book[3:], len(pp)))

    # Почему матч выпал -- видно только отсюда. Две разные причины: имя не
    # опознано norm_team (дыра в _ALIASES) или Oddspedia матч ещё не котирует
    # (getMatchOdds -> 404). Первая чинится, вторая -- нет.
    op = _opener()
    raw = match_list(op)
    print('\nматчей на странице %d, с котировками %d' % (len(raw), len(games)))
    lost = []
    for m in raw:
        h, a = norm_team(m['ht']), norm_team(m['at'])
        if not h or not a:
            bad = ' / '.join(n for n in (m['ht'], m['at']) if not norm_team(n))
            print('   ПРОПУЩЕН %-24s - %-24s  %s  имя не в _ALIASES: %s'
                  % (m['ht'], m['at'], _utc(m['kickoff']), bad))
            lost += [n for n in (m['ht'], m['at']) if not norm_team(n)]
        elif (h, a) not in games:
            print('   БЕЗ ЛИНИИ %-23s - %-24s  %s  getMatchOdds не отдал ничего'
                  % (m['ht'], m['at'], _utc(m['kickoff'])))
    if lost:
        print('чинить синонимом в src/books/__init__.py: %s' % ', '.join(sorted(set(lost))))
    books = sorted({q.book[3:] for q in qs})
    print('конторы (%d): %s' % (len(books), ', '.join(books)))
    print('generated_at: %s' % sorted({q.extra.get('gen') for q in qs}))
