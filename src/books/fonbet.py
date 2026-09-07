# -*- coding: utf-8 -*-
"""
Снятие линии Fonbet (fon.bet) по чемпионату ОАЭ.

ЭНДПОИНТ. Сайт кормится с «ресурсных» зеркал, отдающих всю линию одним JSON
без авторизации, куки и подписи запроса -- достаточно браузерного User-Agent:

    GET https://line01w.bk6bba-resources.com/events/list?lang=en&version=0&scopeMarket=1600

lang=en принципиален: он даёт АНГЛИЙСКИЕ имена команд ('Al Wahda Abu Dhabi'),
а не транслит, и norm_team по ним попадает без отдельного русского словаря.
scopeMarket=1600 тоже обязателен -- без него (и с любым другим значением)
хост отвечает 404, проверено.

Ответ сейчас ~7.8 МБ (раньше был ~1.3 МБ -- контора расширила линию, парсер
от размера не зависит). Внутри нас интересуют три ветки:
    sports[]        -- справочник турниров, ищем id == UAE_SPORT_ID
    events[]        -- матчи, у нужных sportId == UAE_SPORT_ID
    customFactors[] -- котировки, блок на матч: {'e': eventId, 'factors': [...]}

ПОЛНОТА ЛИНИИ: одного запроса ХВАТАЕТ. Есть ещё поштучный эндпоинт

    GET .../events/event?lang=en&version=0&eventId={id}&scopeMarket=1600

и он действительно отдаёт вдвое больше факторов (108-118 против 56-66).
Но проверка по всем пяти матчам тура показала: по НАШИМ рынкам оба источника
совпадают до последней цены (40/40, 40/40, 42/42, 38/38, 34/34 исходов,
расхождений ноль). Лишние полсотни факторов -- это корзины «сколько голов»,
индивидуальные тоталы, точные счета, угловые: всё, что мы и так выбрасываем.
Поэтому по умолчанию контора снимается ОДНИМ запросом, а не шестью: снимки
делаются регулярно, и пятикратный трафик без единой новой котировки не нужен.
Поштучный обход остался под флагом fetch(deep=True) -- на случай, если контора
однажды вынесет альтернативные линии из общего пакета. Симптом, при котором
его стоит включить: в общем пакете у матча остались только 1X2 и основные
тотал с форой (проверяется вызовом fetch(deep=True) и сравнением числа исходов).

ЧТО СЛОМАЕТСЯ ПЕРВЫМ
1. Домен. Он ротируется: был bkfon-resources.com, стал bk6bba-resources.com.
   Когда всё разом начнёт отдавать ConnectionError по всем HOSTS -- открыть
   fon.bet, посмотреть в DevTools -> Network любой запрос /events/list и
   переписать HOSTS. Номера зеркал тоже мрут (line16w/line32w уже мертвы,
   line03w не отвечает), поэтому список перебирается по кругу.
2. id лиги. UAE_SPORT_ID = 13412 ('UAE. Premier League'). При смене сезона
   контора заводит турнир заново с новым id. Найти новый:
       [s for s in resp['sports'] if 'UAE' in (s.get('name') or '')]
   ВНИМАНИЕ: 14542 -- это 'UAE. First Division', второй дивизион. Брать нельзя:
   часть имён там совпадает с премьер-лигой, и мы будем сравнивать цены на
   РАЗНЫЕ матчи. Проверяй по полю name, а не по соседству id.
3. Номера факторов (F_*). Это единственное место, где мы завязаны на
   недокументированную нумерацию конторы; справочника имён рынков на
   ресурсных зеркалах нет (/events/factors и т.п. отдают 404). Разметка ниже
   восстановлена арифметически на живой линии 2026-09-07 (пять матчей тура) и
   перепроверена тремя независимыми тестами -- см. комментарии у каждой группы.
   Если контора перенумерует рынки, неизвестные id просто молча выпадут:
   белый список -- это защита, а не ограничение. Симптом «котировок стало
   вдвое меньше» = пора переснимать разметку тем же способом.

    python src/books/fonbet.py     # сводка по текущему туру
"""
import time

import requests

try:
    from books import Quote, norm_team, sel_total, sel_ah, sel_eh, UA
except ImportError:  # прямой запуск файла: python src/books/fonbet.py
    import os, sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from books import Quote, norm_team, sel_total, sel_ah, sel_eh, UA

BOOK = 'fonbet'

# Зеркала перебираются по порядку. Живыми на 2026-09-07 были 01/02/04/05.
HOSTS = ('line01w.bk6bba-resources.com',
         'line02w.bk6bba-resources.com',
         'line04w.bk6bba-resources.com',
         'line05w.bk6bba-resources.com')

UAE_SPORT_ID = 13412        # 'UAE. Premier League'. НЕ 14542 -- то первый дивизион.
SCOPE = 1600                # единственное принимаемое значение scopeMarket
TIMEOUT = 90
TRIES = 3

# ---------------------------------------------------------------------------
#                         РАЗМЕТКА ФАКТОРОВ (белый список)
# ---------------------------------------------------------------------------
# Ключ -- поле 'f' внутри factors[]; цена -- 'v'; линия -- 'pt' (строка со
# знаком, напр. '-1.5', '2.5') либо 'p' (та же линия × 100).
#
# ВАЖНО ПРО НУМЕРАЦИЮ: id фактора -- это НЕ линия, а СЛОТ. У одного и того же
# f=1730 в матче Al Ain был тотал 1.5, а в первом тайме соседнего матча -- 2.
# Линию всегда читаем из pt/p и никогда не выводим из номера фактора.

F_1, F_X, F_2 = 921, 922, 923
# Двойной шанс. Разметка 924=1X, 1571=12, 925=X2 подтверждена на пяти матчах
# сразу: в симметричном Khor Fakkan -- Baniyas (1X2 = 2.45/3.70/2.45) честные
# цены без маржи равны 1X=1.60, 12=1.33, X2=1.60, а контора даёт
# 924=1.48, 1571=1.23, 925=1.48 -- одно и то же отношение 0.925 у всех трёх.
# Номер 1571 стоит посреди форных слотов (1569/1572), но арифметика однозначна.
F_1X, F_12, F_X2 = 924, 1571, 925

# Тоталы МАТЧА. Лестница проверена на монотонность: цена ТБ растёт с линией,
# ТМ падает, на каждой линии 1/ТБ + 1/ТМ ~ 1.08. Пара 1739/1791 выбивается из
# шага нумерации, но по монотонности это именно ТБ/ТМ на одной линии.
TOT_OVER = {930, 1696, 1727, 1730, 1733, 1736, 1739, 1793, 1796, 1799}
TOT_UNDER = {931, 1697, 1728, 1731, 1734, 1737, 1791, 1794, 1797, 1800}

# ИНДИВИДУАЛЬНЫЕ тоталы команд: 1809-1828 (первой) и 1854-1887 (второй).
# Здесь их нет и быть не должно -- у них те же значения p (50/100/150...), что
# у тоталов матча, и спутать их означало бы сравнивать «тотал Аль-Айна 1.5»
# с «тоталом матча 1.5» у других контор. Отсекаются тем, что не входят в
# белый список выше.

# Форы. Внутри пары номеров первый номер -- всегда первой команде, второй --
# второй, и pt у каждого фактора уже записан со стороны СВОЕЙ команды
# (в Khor Fakkan -- Baniyas у 1672 стоит pt='-1', а у парного 1675 pt='+1').
# Поэтому пары даже не надо сводить: берём знак прямо из pt.
AH_TEAM1 = {910, 927, 989, 1569, 1672, 1677, 1680, 1683}
AH_TEAM2 = {912, 928, 991, 1572, 1675, 1678, 1681, 1684}

# Почему AH, а не EH, даже на целых линиях. Фора Fonbet возвращает ставку при
# точном попадании, то есть азиатская. Доказательство на нулевой форе: в
# Khor Fakkan -- Baniyas (1X2 = 2.45/3.70/2.45) f=927 с pt='0' стоит 1.85.
# Европейская фора 0 = чистая победа, стоила бы как исход '1' (2.45);
# «ничья -- возврат» даёт честные 2.00, с маржой ~1.85. Сходится второе.
# То же на Al Ain (фора 0 при кэфе 1.65 на первого): 1.30 против честных 1.38.
# Поэтому sel_eh здесь не используется -- у конторы в этом разделе (scopeMarket
# 1600) европейских фор нет вовсе. Импорт оставлен по контракту пакета:
# появятся -- добавлять именно сюда, отдельным белым списком.
_ = sel_eh


# ---------------------------------------------------------------------------
#                                  СЕТЬ
# ---------------------------------------------------------------------------
def _get(session, path, tries=TRIES):
    """
    GET по первому живому зеркалу. Перебираем HOSTS, на каждый круг -- пауза.

    Сетевую ошибку НЕ глотаем: если не ответило ни одно зеркало за все круги,
    последнее исключение летит наверх -- сборщик отличит «контора не котирует
    лигу» (пустой список) от «мы не достучались» (исключение).
    """
    last = None
    for attempt in range(tries):
        for host in HOSTS:
            try:
                r = session.get(f'https://{host}{path}', timeout=TIMEOUT)
                r.raise_for_status()
                return r.json()
            except Exception as e:
                last = e
        time.sleep(2 * (attempt + 1))
    raise last


def _session():
    s = requests.Session()
    s.headers.update({'User-Agent': UA,
                      'Accept': 'application/json',
                      'Referer': 'https://fon.bet/'})
    return s


# ---------------------------------------------------------------------------
#                                 РАЗБОР
# ---------------------------------------------------------------------------
def _line(fac):
    """Линия рынка из фактора -> float или None."""
    pt = fac.get('pt')
    if isinstance(pt, str):
        try:
            return float(pt.replace('+', '').strip())
        except ValueError:
            # 'Ничья', '1 and less', '2 or 3', '1-3' -- рынки-корзины,
            # они нам не нужны и в белый список не входят
            return None
    p = fac.get('p')
    # p у корзин -- битовая маска (65536, 131073, 16711684), поэтому
    # доверяем ей только в диапазоне правдоподобных линий
    if isinstance(p, (int, float)) and -2000 <= p <= 2000:
        return p / 100.0
    return None


def _sel(fac):
    """Фактор конторы -> канонический ключ исхода или None, если рынок чужой."""
    f = fac.get('f')
    if f == F_1:
        return '1'
    if f == F_X:
        return 'X'
    if f == F_2:
        return '2'
    if f == F_1X:
        return '1X'
    if f == F_12:
        return '12'
    if f == F_X2:
        return 'X2'
    if f in TOT_OVER or f in TOT_UNDER:
        ln = _line(fac)
        return None if ln is None else sel_total(f in TOT_OVER, ln)
    if f in AH_TEAM1 or f in AH_TEAM2:
        ln = _line(fac)
        return None if ln is None else sel_ah(1 if f in AH_TEAM1 else 2, ln)
    return None


def _factors_of(packet, event_id):
    """Плоский список факторов ИМЕННО этого матча.

    Фильтр по 'e' обязателен: в ответе /events/event рядом лежит блок
    дочернего события (первый тайм) с теми же номерами факторов, но своими
    ценами. Без фильтра тотал тайма уехал бы в тотал матча.
    """
    out = []
    for block in packet.get('customFactors') or []:
        if block.get('e') == event_id:
            out.extend(block.get('factors') or [])
    return out


def _quotes(event, factors, reduced=False):
    """Факторы одного матча -> список Quote."""
    home = norm_team(event.get('team1'))
    away = norm_team(event.get('team2'))
    if not home or not away:
        return []          # чужая команда (молодёжка/кубок) -- молча мимо

    best = {}
    for fac in factors:
        try:
            price = float(fac.get('v'))
        except (TypeError, ValueError):
            continue
        if price <= 1.0:            # заглушки и снятые рынки
            continue
        sel = _sel(fac)
        if sel is None:
            continue
        # Один и тот же исход может прийти из двух слотов (напр. фора -1
        # первой команде живёт то в 927, то в 1672). Берём лучшую цену --
        # ставить всё равно будем по максимуму.
        if price > best.get(sel, 0.0):
            best[sel] = price

    kickoff = event.get('startTime')
    extra = {'event_id': event.get('id')}
    if reduced:
        # Просили deep, но /events/event не ответил и роспись взята из общего
        # пакета. Помечаем, чтобы при разборе полётов было видно, почему у
        # матча вдруг меньше рынков, чем ожидали.
        extra['reduced'] = True
    return [Quote(book=BOOK, home=home, away=away, sel=sel, price=price,
                  kickoff=float(kickoff) if kickoff else None,
                  raw_home=event.get('team1') or '',
                  raw_away=event.get('team2') or '',
                  extra=dict(extra))
            for sel, price in best.items()]


def _events(packet):
    """Матчи лиги ОАЭ из общего пакета.

    level == 1 -- родительское событие (сам матч). У level 2/3 (тайм, десятиминутки)
    team1/team2 = null, их отсекает и norm_team, но лучше не тащить их вовсе.
    place == 'line' -- прематч; лайв нам не нужен, цены там живут секунды.
    """
    return [e for e in (packet.get('events') or [])
            if e.get('sportId') == UAE_SPORT_ID
            and e.get('level') == 1
            and e.get('place') == 'line'
            and e.get('team1') and e.get('team2')]


# ---------------------------------------------------------------------------
#                                  ТОЧКА ВХОДА
# ---------------------------------------------------------------------------
def fetch(deep=False):
    """
    -> list[Quote]. Пустой список = контора сейчас не котирует лигу.

    deep=True добирает роспись поштучно через /events/event. По состоянию на
    2026-09-07 не даёт ни одной новой котировки (см. модульный docstring) --
    держим как страховку на случай, если контора урежет общий пакет.
    """
    s = _session()
    packet = _get(s, f'/events/list?lang=en&version=0&scopeMarket={SCOPE}')

    out = []
    for ev in _events(packet):
        # Чужие команды (молодёжка, кубок) отсеиваем до похода за росписью.
        if not norm_team(ev.get('team1')) or not norm_team(ev.get('team2')):
            continue
        eid = ev['id']
        factors, reduced = _factors_of(packet, eid), False
        if deep:
            try:
                full = _get(s, f'/events/event?lang=en&version=0'
                               f'&eventId={eid}&scopeMarket={SCOPE}')
                factors = _factors_of(full, eid)
            except Exception:
                # /events/event -- ВТОРИЧНЫЙ источник. Падать из-за него нельзя:
                # сборщик получил бы от конторы ноль котировок и потерял вместе
                # с альтернативными линиями ещё и 1X2 с основными тоталом и
                # форой, которые уже лежат в руках. Для проекта, который берёт
                # МАКСИМУМ цены по многим конторам, выпавшая контора дороже
                # урезанной. Настоящая сетевая поломка -- смерть /events/list
                # выше: там _get исключение не глушит, и оно летит наверх.
                reduced = True
        out.extend(_quotes(ev, factors, reduced))
    return out


if __name__ == '__main__':
    import datetime as dt

    qs = fetch()
    games = {}
    for q in qs:
        games.setdefault((q.home, q.away), []).append(q)

    print(f'{BOOK}: матчей {len(games)}, котировок {len(qs)}')
    for (home, away), lst in sorted(games.items(),
                                    key=lambda kv: kv[1][0].kickoff or 0):
        q0 = lst[0]
        when = (dt.datetime.fromtimestamp(q0.kickoff).strftime('%Y-%m-%d %H:%M')
                if q0.kickoff else '?')
        print(f'\n{home} — {away}   {when}   [{q0.raw_home} / {q0.raw_away}]')
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
