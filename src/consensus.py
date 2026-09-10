# -*- coding: utf-8 -*-
"""
КОНСЕНСУС МНОГИХ КОНТОР И ПОИСК ПЕРЕВЕСА ПО ЦЕНЕ.

Зачем не модель. На 717 исходах walk-forward проверено: там, где xG-модель
расходится с рынком, ближе к факту оказывается рынок В КАЖДОМ бакете
расхождения (на краях модель говорит 50.9% при рынке 36.6% и факте 36.4%).
Ставки по перевесу модели дали ROI -23.1%. Поэтому здесь модели нет вообще:
перевес ищется сравнением ЦЕН, а не прогнозом.

Так устроены обе стратегии с опубликованной доходностью:
  * Kaunitz и др. (2017): консенсус 32 контор, ставка там, где МАКСИМАЛЬНАЯ
    цена выше консенсусной справедливой. Прогнозной модели нет вовсе.
    R^2 консенсуса с исходами 0.999 -- он калиброван лучше любой одной конторы.
  * Buchdahl: цена мягкой конторы выше справедливой цены Pinnacle,
    24 150 ставок, +1.81%.

Мы используем ОБА эталона сразу и берём более осторожный. Причина: у каждого
своя слепая зона. Pinnacle точнее всех, но не котирует производные рынки;
консенсус покрывает всё, но его тянут вниз клоны и мягкие конторы.

    python src/consensus.py            # снять всё и показать перевесы
    python src/consensus.py --save     # то же + записать снимок в журнал
"""
import os
import re
import sys
import time
import json
import gzip
import datetime as dt
from collections import defaultdict

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
ROOT = os.path.dirname(HERE)

from books import fetch_all, BETTABLE, Quote, norm_team   # noqa: E402
from markets import DEVIG                             # noqa: E402

SNAPSHOTS = os.path.join(ROOT, 'data', 'odds_snapshots.jsonl.gz')   # старый единый файл
# Снимки пишутся ПО ДНЯМ. Единый gzip рос на ~64 КБ за проход и упёрся бы
# в жёсткий лимит GitHub 100 МБ на файл через ~2 недели, после чего каждый
# push отвергался бы молча. Дневной файл остаётся маленьким и неизменяемым.
SNAPSHOT_DIR = os.path.join(ROOT, 'data', 'snapshots')

# ОСНОВНОЙ метод снятия маржи -- степенной, по калибровке на этой лиге
# (366 матчей, закрывающие цены Bet365, log-loss меньше = лучше):
#   power 0.91675  add 0.91748  oddsratio 0.91843  shin 0.91860  mult 0.92273
# Мультипликативный исключён совсем: он худший по log-loss и сильнее всех
# недооценивает фаворитов (кэф 1.00-1.35: факт 83.6%, mult даёт 72.6%).
# Прежнее правило «минимум по пяти методам» брало для каждого фаворита
# именно mult -- ту оценку, что здесь калибрована хуже всех, -- и на выходе
# давало ноль находок при любой линии. Справедливая цена теперь -- оценка
# основного метода; остальные три служат проверкой устойчивости знака.
DEVIG3 = 'power'
DEVIG2 = 'power'
METHODS = ('power', 'add', 'shin', 'oddsratio')

# Практический пол по коэффициенту. Ниже 1.20 ставка требует огромной суммы
# ради копеечной отдачи, конторы такие ставки режут лимитами, а цена округлена
# так грубо, что один шаг котировки (1.04 -> 1.05) сдвигает матожидание
# на процент. Прежняя версия отбора держала здесь 1.35 по той же причине.
ODDS_FLOOR = 1.20

# Минимальное число контор, чтобы консенсусу вообще можно было верить.
# При двух-трёх конторах медиана -- это не консенсус, а шум.
MIN_BOOKS = 5

# Минимальное число ДОСТУПНЫХ ДЛЯ СТАВКИ операторов, котирующих исход.
# Одинокая линия -- это либо матч, который остальные уже закрыли (и она
# живая или застывшая), либо рынок, который остальные не торгуют. В обоих
# случаях «максимум среди контор» -- не находка, а артефакт. 10 сентября
# ровно так Leon остался единственной конторой через 13 минут после свистка.
MIN_BETTABLE = 2

# Порог перевеса. Стартовое значение, а не истина: 1% -- это измеренный
# разброс между пятью методами снятия маржи на прямых ценах Pinnacle,
# то есть погрешность самого измерения. Ниже неё «перевес» неотличим от
# способа счёта. Уточняется по накопленным снимкам, а не подбирается.
THETA = 0.01


# ---------------------------------------------------------------------------
#                     РАЗБОР ИСХОДОВ НА ЗАМКНУТЫЕ РЫНКИ
# ---------------------------------------------------------------------------
# Семейства одного оператора под разными брендами и через разные агрегаторы.
# Ключ book приходит как 'marathon', 'op:marathonbet', 'fs:Marathonbet.it',
# 'bx:1xbet', 'fs:1xBet.br' -- всё это одна цена, посчитанная один раз.
_CLONE_FAMILIES = {
    '1xbet': ('1xbet', '22bet', 'megapari', '1xstavka', 'linebet'),
    'betano': ('betano', 'stoiximan', 'kaizen'),
    'entain': ('bwin', 'coral', 'ladbrokes', 'sportingbet', 'partypoker', 'betmgm',
               'gamebookers'),
    'mystake': ('mystake', 'goldenbet', 'freshbet', 'jackbit'),
    'fezbet': ('fezbet', 'powbet', 'tooniebet', 'campobet'),
    'bcgame': ('bcgame', 'ninecasino', '4rabet'),
    'williamhill': ('williamhill',),
    'marathon': ('marathon', 'marathonbet'),
    'fonbet': ('fonbet', 'pari'),
    'betcity': ('betcity',),
    'bet365': ('bet365',),
    'unibet': ('unibet', '32red'),
    'betsson': ('betsson', 'betsafe', 'nordicbet'),
    'pinnacle': ('pinnacle',),
}
_STEM = {stem: fam for fam, stems in _CLONE_FAMILIES.items() for stem in stems}


def operator_of(book):
    """'fs:1xBet.br' -> '1xbet'; 'op:marathonbet' -> 'marathon'; 'leon' -> 'leon'."""
    s = (book or '').lower()
    for pre in ('fs:', 'bx:', 'op:'):
        if s.startswith(pre):
            s = s[len(pre):]
    s = re.sub(r'[^a-z0-9]', '', s)
    for stem in sorted(_STEM, key=len, reverse=True):
        if s.startswith(stem):
            return _STEM[stem]
    return s or book


def _parse_sel(sel):
    """'O2.5' -> ('total', 2.5, 'over'); 'AH1-0.25' -> ('ah', -0.25, 1); ..."""
    if sel in ('1', 'X', '2'):
        return ('1x2', None, sel)
    if sel in ('1X', '12', 'X2'):
        return ('dc', None, sel)
    if sel[:1] in 'OU':
        try:
            return ('total', float(sel[1:]), 'over' if sel[0] == 'O' else 'under')
        except ValueError:
            return (None, None, None)
    if sel[:2] in ('AH', 'EH'):
        try:
            team = int(sel[2])
            return ('ah' if sel[:2] == 'AH' else 'eh', float(sel[3:]), team)
        except (ValueError, IndexError):
            return (None, None, None)
    return (None, None, None)


# Минимальная сумма обратных цен, при которой рынок считается ЗАМКНУТЫМ.
# Букмекер всегда закладывает маржу, поэтому у полного набора исходов сумма
# строго больше единицы. Сумма МЕНЬШЕ единицы означает ровно одно: часть
# исходов мы не видим, набор неполный.
#
# Это не теория. Именно так вскрылась дыра на европейской форе: у Marathon
# по матчу Аль-Айн — Аль-Васл пары EH давали суммы 0.876 / 0.930 / 0.988,
# тогда как азиатские форы того же матча -- ровно 1.095. Причина: европейская
# фора ТРЁХСТОРОННЯЯ (победа с форой / НИЧЬЯ с форой / поражение), а мы
# нормировали две цены на единицу. Вероятности раздувались на 14%, и движок
# рисовал перевес +40% на 13 исходах подряд. Порог 1.005 ловит весь этот
# класс ошибок разом, для любого семейства рынков, а не только для EH.
MIN_OVERROUND = 1.005


def _closed(prices):
    """Замкнут ли набор исходов: сумма обратных цен выше единицы с запасом."""
    try:
        return sum(1.0 / float(p) for p in prices) >= MIN_OVERROUND
    except (TypeError, ZeroDivisionError, ValueError):
        return False


def devig_one_book(quotes, m3=None, m2=None):
    """
    Котировки ОДНОЙ конторы на ОДИН матч -> {sel: справедливая вероятность}.

    Снимать маржу можно только с ЗАМКНУТОГО рынка, где исходы образуют полную
    группу: 1X2 целиком, пара тотала на одной линии, пара азиатской форы на
    зеркальных линиях. Одиночная цена без пары непригодна -- из неё нельзя
    вычесть маржу, и попытка это сделать даёт систематическую ошибку
    в свою пользу.

    Европейская фора (EH) НЕ обрабатывается сознательно: у неё три исхода,
    а ничью с форой конторы отдают не всегда и под разными ключами. Пока
    третья нога не приходит от адаптеров, честнее не иметь по ней эталона
    вовсе, чем иметь завышенный.
    """
    m3 = m3 or DEVIG3
    m2 = m2 or DEVIG2
    by = {q.sel: q.price for q in quotes}
    out = {}

    if all(k in by for k in ('1', 'X', '2')):
        prices = [by['1'], by['X'], by['2']]
        if _closed(prices):
            q = DEVIG[m3](prices)
            if not any(np.isnan(q)):
                out['1'], out['X'], out['2'] = float(q[0]), float(q[1]), float(q[2])

    tot = defaultdict(dict)
    for sel, price in by.items():
        kind, line, side = _parse_sel(sel)
        if kind == 'total':
            tot[line][side] = price

    for line, sides in tot.items():
        if 'over' in sides and 'under' in sides and _closed(sides.values()):
            q = DEVIG[m2]([sides['over'], sides['under']])
            if not any(np.isnan(q)):
                out[f'O{line:g}'] = float(q[0])
                out[f'U{line:g}'] = float(q[1])

    # Азиатская фора: 'AH1-0.5' замыкается с 'AH2+0.5'. Ключ группировки --
    # линия СО СТОРОНЫ ХОЗЯЕВ (для гостей знак переворачивается). Группировка
    # по модулю линии была ошибкой: у конторы, котирующей и -0.25, и +0.25,
    # вторая пара затирала первую, а уцелевшие ноги не были зеркальными --
    # у 1xbet из 17 замкнутых линий выживало 13, и терялись как раз линии
    # около нуля, самые ликвидные.
    pairs = defaultdict(dict)
    for sel, price in by.items():
        kind, line, team = _parse_sel(sel)
        if kind == 'ah':
            home_line = line if team == 1 else -line
            pairs[round(home_line, 4)][team] = (price, line)
    for _hl, sides in pairs.items():
        if 1 not in sides or 2 not in sides:
            continue
        (p1, l1), (p2, l2) = sides[1], sides[2]
        if abs(l1 + l2) > 1e-9:              # линии обязаны быть зеркальными
            continue
        if not _closed([p1, p2]):
            continue
        q = DEVIG[m2]([p1, p2])
        if not any(np.isnan(q)):
            out[f'AH1{"+" if l1 >= 0 else "-"}{abs(l1):g}'] = float(q[0])
            out[f'AH2{"+" if l2 >= 0 else "-"}{abs(l2):g}'] = float(q[1])
    return out


def build_consensus(quotes, methods=METHODS):
    """
    Все котировки -> {(home, away): {sel: {'p': ОСТОРОЖНАЯ оценка, 'n': контор,
                                           'p_by': {метод: медиана},
                                           'spread': разброс между конторами}}}

    Два усреднения, и оба нужны.

    По конторам берётся МЕДИАНА: клоны одного оператора (1xbet=22bet=megapari
    и подобные) и просто кривые цены не должны тянуть оценку. Медиана к ним
    устойчива, среднее -- нет.

    По методам снятия маржи берётся МИНИМУМ, то есть самая осторожная оценка
    вероятности (самая высокая справедливая цена). Это не перестраховка,
    а единственный честный ответ на измеренный факт: расхождение между
    методами -- реальная неопределённость измерения, а не выбор вкуса.
    Замер на живой линии 7 сентября 2026: при базовом методе движок нашёл
    21 «перевес», и ВСЕ 21 меняли знак при смене метода -- положительны
    только под степенным, отрицательны под мультипликативным и аддитивным.
    Все они лежали на кэфах ниже 1.35, где степенной де-виг сдвигает массу
    к фавориту сильнее прочих. Это был бы 21 проигрышный сигнал подряд.
    """
    # Группируем по ОПЕРАТОРУ, а не по строке book. Один оператор приходит
    # под несколькими ключами (marathon, op:marathonbet, fs:Marathonbet.it;
    # 1xbet = 22bet = megapari через три агрегатора) и раньше считался
    # несколькими независимыми конторами: на снимке 7 сентября 374 из 452
    # ячеек проходили MIN_BOOKS=5 силами одного семейства 1xBet. Ключ
    # прямой конторы (где ставим) имеет приоритет, дальше -- первый встречный.
    ordered = sorted(quotes, key=lambda q: 0 if q.book in BETTABLE else 1)
    by_match_op = defaultdict(lambda: defaultdict(dict))
    for q in ordered:
        d = by_match_op[(q.home, q.away)][operator_of(q.book)]
        if q.sel not in d:
            d[q.sel] = q
    by_match_book = {m: {op: list(d.values()) for op, d in ops.items()}
                     for m, ops in by_match_op.items()}

    out = {}
    for match, books in by_match_book.items():
        per_method = {m: defaultdict(list) for m in methods}
        for book, qs in books.items():
            for m in methods:
                for sel, p in devig_one_book(qs, m3=m, m2=m).items():
                    if 0.001 < p < 0.999:
                        per_method[m][sel].append(p)
        sels = set().union(*[set(d) for d in per_method.values()]) if per_method else set()
        agg = {}
        for sel in sels:
            med, ns = {}, []
            for m in methods:
                vals = per_method[m].get(sel) or []
                if len(vals) >= MIN_BOOKS:
                    med[m] = float(np.median(vals))
                    ns.append(len(vals))
            if len(med) < len(methods):      # метод не смог -- нет и оценки
                continue
            # Разброс между конторами считаем по первому методу набора: набор
            # может не содержать DEVIG2, и жёсткая ссылка на него роняла расчёт.
            base = per_method[methods[0]].get(sel) or []
            agg[sel] = dict(p=med[methods[0]], p_by=med, n=max(ns),
                            spread=float(max(base) - min(base)) if base else None,
                            method_spread=float(max(med.values()) - min(med.values())))
        if agg:
            out[match] = agg
    return out


# ---------------------------------------------------------------------------
#                              ПОИСК ПЕРЕВЕСА
# ---------------------------------------------------------------------------
def find_value(quotes, consensus, pin_fair=None, theta=THETA, bettable=BETTABLE):
    """
    -> список находок, отсортированный по перевесу.

    Правило: берём МАКСИМАЛЬНУЮ цену среди контор, где можно поставить,
    и сравниваем со справедливой ценой. Справедливая -- более осторожная
    из двух: консенсуса многих контор и прямой цены Pinnacle.

    Осторожная = дающая МЕНЬШУЮ вероятность, то есть более высокую
    справедливую цену. Так порог труднее пройти случайно: чтобы ставка
    прошла, она должна быть выгодна против ОБОИХ эталонов сразу.
    """
    best = defaultdict(lambda: (0.0, None))
    ko = {}
    n_bet = defaultdict(set)
    for q in quotes:
        if q.kickoff:
            ko[(q.home, q.away)] = min(ko.get((q.home, q.away), q.kickoff), q.kickoff)
        if q.book not in bettable or q.price < ODDS_FLOOR:
            continue
        k = (q.home, q.away, q.sel)
        n_bet[k].add(operator_of(q.book))
        if q.price > best[k][0]:
            best[k] = (q.price, q.book)

    out = []
    for (h, a, sel), (price, book) in best.items():
        if len(n_bet[(h, a, sel)]) < MIN_BETTABLE:
            continue                         # одинокая линия -- см. MIN_BETTABLE
        c = (consensus.get((h, a)) or {}).get(sel)
        pf = (pin_fair.get((h, a)) or {}).get(sel) if pin_fair else None
        if c is None and pf is None:
            continue
        cands = [x for x in (c['p'] if c else None, pf) if x is not None]
        p_fair = min(cands)                    # осторожная оценка
        src = ('консенсус' if (c and p_fair == c['p']) else 'Pinnacle')
        ev = price * p_fair - 1.0
        # Матожидание при КАЖДОМ методе снятия маржи. Находка засчитывается,
        # только если положительна при всех: расхождение методов -- это
        # погрешность измерения, и «перевес», живущий лишь при одном из них,
        # неотличим от способа счёта. На живой линии этот фильтр снял
        # 21 находку из 21, все на кэфах ниже 1.35.
        ev_by = {}
        if c:
            for meth, pm in (c.get('p_by') or {}).items():
                ev_by[meth] = price * min([pm] + ([pf] if pf is not None else [])) - 1.0
        # Без консенсуса проверить устойчивость нечем: pinnacle_fair() считает
        # одним методом. Такие исходы показываем, но НЕ засчитываем -- иначе
        # в сигнал просочится ровно та ошибка, ради которой фильтр и заводился.
        ev_worst = min(ev_by.values()) if ev_by else ev
        robust = bool(ev_by)
        out.append(dict(
            home=h, away=a, sel=sel, price=price, book=book, kickoff=ko.get((h, a)),
            p_fair=p_fair, fair_price=1.0 / p_fair, ev=ev, ev_worst=ev_worst,
            ev_by=ev_by, ref=src,
            n_books=(c['n'] if c else 0),
            spread=(c['spread'] if c else None),
            p_cons=(c['p'] if c else None), p_pin=pf,
            hit=(robust and ev_worst > theta)))
    return sorted(out, key=lambda r: -r['ev_worst'])


# ---------------------------------------------------------------------------
#                                 ЖУРНАЛ
# ---------------------------------------------------------------------------
def save_snapshot(quotes, errs):
    """
    Пишем СЫРЫЕ котировки, а не выводы. Пороги и методы снятия маржи ещё
    будут меняться; переоценить их задним числом можно только по сырым
    ценам. Один снимок -- одна строка gzip-jsonl.
    """
    rec = dict(
        at=dt.datetime.now(dt.timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ'),
        errors=errs,
        quotes=[dict(b=q.book, h=q.home, a=q.away, s=q.sel, p=q.price,
                     k=q.kickoff) for q in quotes],
    )
    os.makedirs(SNAPSHOT_DIR, exist_ok=True)
    path = os.path.join(SNAPSHOT_DIR, rec['at'][:10] + '.jsonl.gz')
    with gzip.open(path, 'at', encoding='utf-8') as f:
        f.write(json.dumps(rec, ensure_ascii=False) + '\n')
    return len(rec['quotes']), path


def pinnacle_fair():
    """Прямые справедливые вероятности Pinnacle -> {(home, away): {sel: p}}."""
    # Имена -- через тот же norm_team, что и у всех контор. Раньше здесь был
    # NAME_MAP из sharp.py, в котором нет 'Sharjah FC' (именно так Pinnacle
    # зовёт Шарджу): ключ не совпадал с адаптерами, и эталон Pinnacle для
    # любого матча Шарджи молча пропадал -- в журнале это видно как ярус «?».
    # Ошибки не глотаем: молчащий эталон неотличим от отсутствующего.
    try:
        import pinnacle
        games = pinnacle.parse()
    except Exception as e:
        print(f'  Pinnacle недоступен: {type(e).__name__}: {e}', file=sys.stderr)
        return {}
    out = {}
    for g in games.values():
        h, a = norm_team(g.get('home')), norm_team(g.get('away'))
        if not h or not a:
            print(f"  Pinnacle: не распознал {g.get('home')!r} — {g.get('away')!r}",
                  file=sys.stderr)
            continue
        d = {}
        ml = g.get('moneyline') or {}
        if all(k in ml for k in ('home', 'draw', 'away')):
            q = DEVIG[DEVIG3]([ml['home'], ml['draw'], ml['away']])
            if not any(np.isnan(q)):
                d['1'], d['X'], d['2'] = float(q[0]), float(q[1]), float(q[2])
        for L, v in (g.get('totals') or {}).items():
            if 'over' in v and 'under' in v:
                q = DEVIG[DEVIG2]([v['over'], v['under']])
                if not any(np.isnan(q)):
                    d[f'O{float(L):g}'] = float(q[0])
                    d[f'U{float(L):g}'] = float(q[1])
        for L, v in (g.get('spreads') or {}).items():
            if 'home' in v and 'away' in v:
                q = DEVIG[DEVIG2]([v['home'], v['away']])
                if not any(np.isnan(q)):
                    L = float(L)
                    d[f'AH1{"+" if L >= 0 else "-"}{abs(L):g}'] = float(q[0])
                    d[f'AH2{"+" if -L >= 0 else "-"}{abs(L):g}'] = float(q[1])
        if d:
            out[(h, a)] = d
    return out


def main():
    t0 = time.time()
    print('Снимаю линии...')
    quotes, errs = fetch_all()
    print(f'\nвсего котировок: {len(quotes)}, за {time.time()-t0:.0f} с')
    if errs:
        print('ошибки:', ', '.join(f'{k}' for k in errs))

    books_ok = sorted({q.book for q in quotes})
    bet_ok = sorted({q.book for q in quotes if q.book in BETTABLE})
    print(f'источников с данными: {len(books_ok)}, из них доступных для ставки: {len(bet_ok)}')
    print(f'  ставить можно в: {", ".join(bet_ok) or "нет"}')

    cons = build_consensus(quotes)
    pin = pinnacle_fair()
    print(f'матчей в консенсусе: {len(cons)}, у Pinnacle: {len(pin)}')

    val = find_value(quotes, cons, pin)
    hits = [v for v in val if v['hit']]
    print(f'\nисходов с ценой: {len(val)}, прошли порог +{100*THETA:.1f}%: {len(hits)}')
    print()
    if not val:
        print('нечего показывать')
        return
    print('%-32s %-10s %6s %-9s %8s %9s %9s %4s' %
          ('матч', 'исход', 'кэф', 'контора', 'справ.', 'EV', 'EV худш.', 'кнт'))
    for v in val[:25]:
        mark = '  <<<' if v['hit'] else ''
        print('%-32s %-10s %6.2f %-9s %8.3f %+8.2f%% %+8.2f%% %4d%s' % (
            f"{v['home'][:14]} — {v['away'][:14]}", v['sel'], v['price'],
            v['book'][:9], v['fair_price'], 100 * v['ev'],
            100 * v['ev_worst'], v['n_books'], mark))

    if '--save' in sys.argv:
        n, path = save_snapshot(quotes, errs)
        print(f'\nснимок записан: {n} котировок -> {path}')


if __name__ == '__main__':
    main()
