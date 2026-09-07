# -*- coding: utf-8 -*-
"""
Линия Baltbet (baltbet.ru) по чемпионату ОАЭ.

Контора отдаёт публичный REST JSON -- тот же, что ест её собственный фронт,
без ключей и без авторизации. Единственная нетривиальная деталь: ОБЯЗАТЕЛЕН
заголовок `bb_appSource: UniRu`. Без него сервер отвечает 400 с ПУСТЫМ телом,
что при отладке выглядит как «сломался парсер», хотя сломан запрос.

Берём линию в два захода, потому что это два разных по смыслу эндпоинта:

  1) POST /api/prematch/leagues  {"ids":[120656]}
     Роспись лиги: список матчей (id, team1, team2, startTime) плюс ОДНА
     главная линия на каждый рынок -- исход, двойной шанс, одна фора,
     один тотал. Дёшево (~5 КБ) и всегда достаточно, чтобы знать состав тура.

  2) GET /api/grouping/events?ids=..&ids=..
     Полная роспись матча: все альтернативные форы и тоталы (в поле
     additionalCoefsCount видно, что их полторы сотни). Отсюда мы берём
     основную массу котировок.

Первый заход остаётся страховкой второго: главные цены дублируются в обоих
ответах, поэтому если grouping когда-нибудь сменит схему, из leagues всё
равно приедет 1X2, двойной шанс и по одной линии форы/тотала.

ПОЧЕМУ ФОРА -- АЗИАТСКАЯ, А НЕ ЕВРОПЕЙСКАЯ. Рынок «Фора» у Baltbet двусторонний
(Ф1к/Ф2к, ничьей как отдельного исхода нет), и целые линии в нём тоже азиатские
-- на них ничья/точное попадание в линию даёт возврат. Проверено арифметикой на
живой линии: Аль Айн 1X2 = 1.63/4.20/4.20, маржа 1.0897; честный draw-no-bet с
той же маржой даёт 1.274, а в линии «Ф1 0» стоит ровно 1.28. Европейская фора
была бы трёхисходным рынком и стоила бы иначе. Поэтому ВСЕ линии этого рынка,
включая 0, -1, -2, уходят через sel_ah, и ни одна -- через sel_eh.

ЧТО СЛОМАЕТСЯ ПРИ СМЕНЕ СЕЗОНА. Id лиги 120656 живёт ровно один сезон: летом
контора заводит турнир заново и номер меняется. Симптом -- пустой список
событий при рабочем HTTP 200. Новый id ищется так:

    python src/books/baltbet.py --find-league

(GET /api/prematch/sport/17/events?offset=all -- вся футбольная роспись,
около 1.7 МБ, sport id 17 = футбол; ищем title «ОАЭ. Чемпионат»).

Второе, что может поехать, -- id рынков (12629/12630/12631/12632) и smid
(46..49). Они привязаны к виду спорта, а не к сезону, и меняются редко, но
если счётчик котировок вдруг упал до нуля при живых матчах -- смотреть надо
именно туда, распечатав markets[].name из grouping.

    python src/books/baltbet.py            # сводка по текущей линии
    python src/books/baltbet.py --find-league
"""
import os
import re
import sys
import time

import requests

# Запуск файла напрямую (python src/books/baltbet.py) кладёт в sys.path саму
# папку books, а не src, и пакет `books` становится не виден. При импорте
# сборщиком (`import books.baltbet`) __package__ непустой и ветка не трогается.
if __package__ in (None, ''):
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# sel_eh импортируется по общему контракту, но здесь намеренно не используется:
# рынок «Фора» у Baltbet азиатский на всех линиях, включая целые (см. docstring).
from books import Quote, norm_team, sel_total, sel_ah, sel_eh, UA  # noqa: F401

BOOK = 'baltbet'

# Id лиги на сезон 2026/27. Как перевыпустить -- см. модульный docstring.
LEAGUE_ID = 120656
SPORT_FOOTBALL = 17

_API = 'https://events.baltbet.ru/api'
# Без bb_appSource сервер отдаёт 400 с пустым телом -- это не опечатка.
_HDRS = {
    'bb_appSource': 'UniRu',
    'User-Agent': UA,
    'Accept': 'application/json',
    'Content-Type': 'application/json',
    'Origin': 'https://baltbet.ru',
    'Referer': 'https://baltbet.ru/',
}

# --- схема короткой росписи (prematch/leagues) -----------------------------
# smid -- id «супер-рынка», smcid -- id колонки внутри него.
_SM_RESULT, _SM_DCHANCE, _SM_HANDICAP, _SM_TOTAL = 46, 47, 48, 49
_SMC_1, _SMC_X, _SMC_2 = 715, 716, 717
_SMC_1X, _SMC_12, _SMC_X2 = 718, 719, 720
_SMC_H1_LINE, _SMC_H1_ODD, _SMC_H2_LINE, _SMC_H2_ODD = 721, 722, 723, 724
_SMC_T_LINE, _SMC_T_OVER, _SMC_T_UNDER = 725, 726, 727

# --- схема полной росписи (grouping/events) --------------------------------
# Только рынки на ВЕСЬ матч. Тайминговые (12637 «Фора в тайме», 12663 «Тотал
# в тайме») и индивидуальные тоталы (12654/12692) сюда сознательно не входят:
# их структура один в один как у основных, и по невнимательности они легко
# подмешиваются в тотал матча.
_MK_RESULT, _MK_DCHANCE, _MK_HANDICAP, _MK_TOTAL = 12629, 12630, 12631, 12632

_KIND_RESULT = {'TEAM1': '1', 'DRAW': 'X', 'TEAM2': '2'}
_KIND_DCHANCE = {'TEAM1_OR_DRAW': '1X', 'NOT_DRAW': '12', 'TEAM2_OR_DRAW': 'X2'}

# Маркеры не-основных составов. Нужны, потому что общий norm_team() режет
# скобки как пунктуацию и по подстроке узнаёт в «Аджман(23)» взрослый Ajman
# Club (проверено на живой лиге 1321563 «ОАЭ. Чемпионат до 23 лет»: туда
# провалились Аджман(23), Аль Наср Дубай(23), Аль Вахда Абу-Даби(23)).
# Сегодня нас это не задевает -- мы тянем лигу по id, и молодёжка лежит в
# другой. Но стоит конторе подмешать юниорский тур в основной турнир, и мы
# молча выдадим цены не того матча, а это худший из возможных сбоев.
# Словарь в books/__init__.py -- общий для всех адаптеров, править его отсюда
# нельзя, поэтому подстраховываемся у себя.
_YOUTH = re.compile(
    r'\(\s*\d{2}\s*\)'              # (23), (19), (21)
    r'|\bu-?\d{2}\b'                # U23, U-19
    r'|молод|дубл|резерв|\bмол\b'   # молодёжный / дубль / резерв
    r'|\(\s*жен|женщ',              # женские команды
    re.IGNORECASE)

_TIMEOUT = 40
_TRIES = 3
# grouping ест несколько id за раз, но городить километровый URL незачем.
_CHUNK = 10


# ---------------------------------------------------------------------------
#                                 ТРАНСПОРТ
# ---------------------------------------------------------------------------
def _request(method, url, **kw):
    """
    Сетевые ошибки НЕ глушим -- сборщик обязан отличить «контора не котирует
    лигу» от «мы сломались». Ретраим только сам обрыв связи.
    """
    last = None
    for i in range(_TRIES):
        try:
            r = requests.request(method, url, headers=_HDRS, timeout=_TIMEOUT, **kw)
            r.raise_for_status()
            return r.json()
        except Exception as e:
            last = e
            if i == _TRIES - 1:
                raise
            time.sleep(1.5 * (i + 1))
    raise last


def _num(v):
    """
    Коэффициент -> float или None.

    Отдельная функция, потому что мусора в поле хватает: приостановленный
    исход приезжает как None/0, а иногда как строка.
    """
    try:
        x = float(v)
    except (TypeError, ValueError):
        return None
    # кэф ниже 1.01 -- это не цена, а заглушка снятого исхода
    return x if x > 1.01 else None


def _line(text):
    """
    Текст линии -> float или None.

    ГЛАВНАЯ ГРАБЛЯ ИСТОЧНИКА: десятичный разделитель -- ЗАПЯТАЯ ("2,5",
    "-1,5"). float("2,5") падает, и без этой замены тоталы молча исчезают.
    Плюс к тому линия форы приезжает со знаком и иногда с пробелами: "+1,5".
    """
    if text is None:
        return None
    s = str(text).strip().replace(',', '.').replace(' ', '').replace('−', '-')
    if not s or s in ('+', '-'):
        return None
    if s.startswith('+'):
        s = s[1:]
    try:
        return float(s)
    except ValueError:
        return None


# ---------------------------------------------------------------------------
#                          НАКОПИТЕЛЬ КОТИРОВОК
# ---------------------------------------------------------------------------
class _Basket:
    """
    Копилка с дедупликацией по (матч, исход).

    Одна и та же цена приезжает дважды -- из короткой росписи и из полной,
    -- а внутри полной главная линия дублируется отдельной колонкой (ctid 26
    и ctid 522 у форы). Держим лучшую цену: у одной конторы дубли исхода
    должны совпадать, а если разошлись -- поставить можно по большей.
    """

    def __init__(self):
        self._q = {}

    def add(self, m, sel, price):
        price = _num(price)
        if not sel or price is None:
            return
        key = (m['home'], m['away'], sel)
        old = self._q.get(key)
        if old is not None and old.price >= price:
            return
        self._q[key] = Quote(
            book=BOOK, home=m['home'], away=m['away'], sel=sel, price=price,
            kickoff=m['kickoff'], raw_home=m['raw_home'], raw_away=m['raw_away'],
        )

    def result(self):
        return sorted(self._q.values(), key=lambda q: (q.kickoff or 0, q.home, q.sel))


# ---------------------------------------------------------------------------
#                     РАЗБОР КОРОТКОЙ РОСПИСИ (prematch/leagues)
# ---------------------------------------------------------------------------
def _match_index(payload):
    """
    Ответ leagues -> {event_id: карточка матча}.

    Матч с неопознанной командой ПРОПУСКАЕМ молча: в разделе конторы рядом с
    основой лежат молодёжные и кубковые составы, и угадывать имя самим нельзя
    -- ценой ошибки будет сравнение цен на разные матчи.
    """
    out, skipped = {}, []
    for sport in payload or []:
        for lg in sport.get('leagues') or []:
            for ev in lg.get('events') or []:
                raw_h = (ev.get('team1') or '').strip()
                raw_a = (ev.get('team2') or '').strip()
                if _YOUTH.search(raw_h) or _YOUTH.search(raw_a):
                    if raw_h and raw_a:
                        skipped.append(f'{raw_h} - {raw_a} (не основной состав)')
                    continue
                home, away = norm_team(raw_h), norm_team(raw_a)
                if not home or not away or home == away:
                    if raw_h and raw_a:
                        skipped.append(f'{raw_h} - {raw_a}')
                    continue
                eid = ev.get('id')
                if eid is None:
                    continue
                ts = ev.get('startTime')
                out[eid] = dict(home=home, away=away, raw_home=raw_h, raw_away=raw_a,
                                kickoff=float(ts) if ts else None,
                                league=lg.get('title', ''), markets=ev.get('markets') or [])
    return out, skipped


def _parse_short(m, basket):
    """Главные цены из росписи лиги -- страховка на случай поломки grouping."""
    for mk in m.get('markets') or []:
        smid = mk.get('smid')
        # {smcid: значение} -- в коротком ответе цена лежит в value,
        # а текст линии в text, поэтому забираем оба поля.
        vals = {c.get('smcid'): c.get('value') for c in mk.get('coefs') or []}
        text = {c.get('smcid'): c.get('text') for c in mk.get('coefs') or []}

        if smid == _SM_RESULT:
            basket.add(m, '1', vals.get(_SMC_1))
            basket.add(m, 'X', vals.get(_SMC_X))
            basket.add(m, '2', vals.get(_SMC_2))
        elif smid == _SM_DCHANCE:
            basket.add(m, '1X', vals.get(_SMC_1X))
            basket.add(m, '12', vals.get(_SMC_12))
            basket.add(m, 'X2', vals.get(_SMC_X2))
        elif smid == _SM_HANDICAP:
            # линия уже записана со стороны своей команды: Ф1 "-1", Ф2 "+1"
            h1, h2 = _line(text.get(_SMC_H1_LINE)), _line(text.get(_SMC_H2_LINE))
            if h1 is not None:
                basket.add(m, sel_ah(1, h1), vals.get(_SMC_H1_ODD))
            if h2 is not None:
                basket.add(m, sel_ah(2, h2), vals.get(_SMC_H2_ODD))
        elif smid == _SM_TOTAL:
            ln = _line(text.get(_SMC_T_LINE))
            if ln is not None:
                basket.add(m, sel_total(True, ln), vals.get(_SMC_T_OVER))
                basket.add(m, sel_total(False, ln), vals.get(_SMC_T_UNDER))


# ---------------------------------------------------------------------------
#                     РАЗБОР ПОЛНОЙ РОСПИСИ (grouping/events)
# ---------------------------------------------------------------------------
def _outcome_line(outcome):
    """
    Линия исхода из params -> (значение, это_тайм).

    В params лежит смесь: собственно линия ("-1,5"), пустая болванка
    (ctid 520 с value "") и, у тайминговых рынков, параметр kind=="ROUND"
    со значением «1-й тайм». Матч и тайм в такой схеме различаются ТОЛЬКО
    этим признаком, поэтому ROUND -- стоп-сигнал: такой исход не наш.
    """
    val = None
    for p in outcome.get('params') or []:
        if (p.get('kind') or '').upper() == 'ROUND':
            return None, True
        if val is None:
            val = _line(p.get('value'))
    return val, False


def _parse_full(ev, index, basket):
    """Все линии одного матча из полной росписи."""
    m = index.get(ev.get('id'))
    if not m:
        return
    for mk in ev.get('markets') or []:
        mid = mk.get('id')
        if mid not in (_MK_RESULT, _MK_DCHANCE, _MK_HANDICAP, _MK_TOTAL):
            continue
        for o in mk.get('outcomes') or []:
            kind = (o.get('kind') or '').upper()
            price = o.get('value')
            line, is_period = _outcome_line(o)
            if is_period:
                continue

            if mid == _MK_RESULT:
                basket.add(m, _KIND_RESULT.get(kind), price)
            elif mid == _MK_DCHANCE:
                basket.add(m, _KIND_DCHANCE.get(kind), price)
            elif mid == _MK_HANDICAP:
                if line is None:
                    continue
                # знак в params уже со стороны своей команды: Ф1 "-1,5", Ф2 "+1,5"
                if kind == 'TEAM1':
                    basket.add(m, sel_ah(1, line), price)
                elif kind == 'TEAM2':
                    basket.add(m, sel_ah(2, line), price)
            elif mid == _MK_TOTAL:
                if line is None:
                    continue
                if kind == 'OVER':
                    basket.add(m, sel_total(True, line), price)
                elif kind == 'UNDER':
                    basket.add(m, sel_total(False, line), price)


# ---------------------------------------------------------------------------
#                              ПУБЛИЧНАЯ ТОЧКА ВХОДА
# ---------------------------------------------------------------------------
def fetch():
    """-> list[Quote] по чемпионату ОАЭ. Пустой список = контора не котирует лигу."""
    payload = _request('POST', f'{_API}/prematch/leagues', json={'ids': [LEAGUE_ID]})
    index, _ = _match_index(payload)
    if not index:
        return []

    basket = _Basket()
    for m in index.values():
        _parse_short(m, basket)

    ids = list(index)
    for i in range(0, len(ids), _CHUNK):
        chunk = ids[i:i + _CHUNK]
        full = _request('GET', f'{_API}/grouping/events',
                        params=[('ids', e) for e in chunk])
        for ev in full or []:
            _parse_full(ev, index, basket)

    return basket.result()


# ---------------------------------------------------------------------------
#                          ДИАГНОСТИКА (только из консоли)
# ---------------------------------------------------------------------------
def _find_league():
    """
    Переоткрыть id лиги, когда сезон сменился и 120656 отдаёт пустоту.
    Тянет всю футбольную роспись (~1.7 МБ), поэтому в fetch() ей не место.
    """
    data = _request('GET', f'{_API}/prematch/sport/{SPORT_FOOTBALL}/events',
                    params={'offset': 'all'})
    hits = []

    def walk(node):
        if isinstance(node, dict):
            t = node.get('title') or node.get('name') or ''
            if node.get('id') and isinstance(t, str) and 'оаэ' in t.lower():
                hits.append((node['id'], t, len(node.get('events') or [])))
            for v in node.values():
                walk(v)
        elif isinstance(node, list):
            for v in node:
                walk(v)

    walk(data)
    return sorted(set(hits))


def _main():
    if hasattr(sys.stdout, 'reconfigure'):          # кириллица в консоли Windows
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')

    if '--find-league' in sys.argv:
        print('Турниры с «ОАЭ» в названии (id, заголовок, матчей):')
        for i, t, n in _find_league():
            print(f'  {i:>8}  {t}  -- матчей {n}')
        return

    payload = _request('POST', f'{_API}/prematch/leagues', json={'ids': [LEAGUE_ID]})
    index, skipped = _match_index(payload)
    raw_total = sum(len(lg.get('events') or [])
                    for sp in payload or [] for lg in sp.get('leagues') or [])
    print(f'BALTBET  лига {LEAGUE_ID}: матчей в росписи {raw_total}, опознано {len(index)}')
    if skipped:
        print('  пропущены (norm_team не знает имени): ' + '; '.join(skipped))

    qs = fetch()
    if not qs:
        print('  котировок нет -- контора сейчас не котирует лигу (между турами)')
        return

    by_match = {}
    for q in qs:
        by_match.setdefault((q.kickoff, q.home, q.away), []).append(q)

    for (ts, home, away), items in sorted(by_match.items(), key=lambda kv: kv[0][0] or 0):
        when = time.strftime('%Y-%m-%d %H:%M', time.localtime(ts)) if ts else '??'
        sels = {q.sel: q.price for q in items}
        raw = next(iter(items))
        print(f'\n{home} - {away}   {when}   [{raw.raw_home} - {raw.raw_away}]')
        print('  исход  ' + '  '.join(f'{k} {sels[k]:.2f}'
                                      for k in ('1', 'X', '2') if k in sels))
        print('  дв.шанс ' + '  '.join(f'{k} {sels[k]:.2f}'
                                       for k in ('1X', '12', 'X2') if k in sels))
        tot = sorted((k for k in sels if k[0] in 'OU'), key=lambda k: (float(k[1:]), k))
        print('  тоталы  ' + '  '.join(f'{k} {sels[k]:.2f}' for k in tot))
        ah = sorted((k for k in sels if k.startswith('AH')),
                    key=lambda k: (k[2], float(k[3:])))
        print('  форы    ' + '  '.join(f'{k} {sels[k]:.2f}' for k in ah))
        print(f'  всего исходов: {len(items)}')

    fams = dict(
        итог=sum(1 for q in qs if q.sel in ('1', 'X', '2')),
        дв_шанс=sum(1 for q in qs if q.sel in ('1X', '12', 'X2')),
        тотал=sum(1 for q in qs if q.sel[0] in 'OU'),
        фора=sum(1 for q in qs if q.sel.startswith('AH')),
        евро_фора=sum(1 for q in qs if q.sel.startswith('EH')),
    )
    print(f'\nИТОГО: матчей {len(by_match)}, котировок {len(qs)}   ' +
          '  '.join(f'{k}={v}' for k, v in fams.items()))


if __name__ == '__main__':
    _main()
