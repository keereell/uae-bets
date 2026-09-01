# -*- coding: utf-8 -*-
"""
Сопоставление рынков БЕТСИТИ с вероятностями модели.

Возвращает для каждого исхода (p_win, p_push, p_lose).
Целые линии тоталов и фор у БЕТСИТИ считаются азиатскими (при точном
попадании — возврат ставки); это подтверждается тем, что маржа на целых
линиях совпадает с маржой на половинных (~8%).
"""
import re
import numpy as np
from markets import (wdl, double_chance, totals, team_total, btts, odd_even,
                     asian_handicap, asian_total, interval_goals, _grid)

NUM = r'([+-]?\d+(?:[.,]\d+)?)'


def _f(s):
    return float(str(s).replace(',', '.'))


def _pure(p):
    """Обычный исход без возврата."""
    return (float(p), 0.0, float(1 - p))


def price_bet(M, market, line, outcome, home_name='', away_name=''):
    """-> (p_win, p_push, p_lose) либо None, если рынок не поддержан."""
    mk = (market or '').strip()
    ln = (line or '').strip()
    oc = (outcome or '').strip()
    low = mk.lower()

    # исключаем таймы, интервалы и всё, что модель полного матча не считает
    bad = ('тайм', 'мин', 'интервал', 'по ходу', 'время', 'подряд', 'первый гол',
           'последний гол', 'вести в счёте', 'вести в счете', 'волевая',
           'будет проигрывать', 'забьет первый гол', 'суммарный тотал минут')
    if any(b in low for b in bad) or any(b in oc.lower() for b in bad):
        return None

    p = wdl(M)

    # ---------------- 1X2
    if mk in ('1X2', 'Исход') or (mk == '' and oc in ('1', 'X', '2')):
        return _pure({'1': p['H'], 'X': p['D'], '2': p['A']}[oc]) if oc in ('1', 'X', '2') else None

    # ---------------- двойной исход
    if 'двойной исход' in low:
        dc = double_chance(M)
        return _pure(dc[oc]) if oc in dc else None

    # ---------------- тотал матча
    if mk in ('Тотал', 'Азиатский тотал') or low.startswith('доп. тотал'):
        m = re.search(NUM, ln)
        if not m:
            return None
        L = _f(m.group(1))
        if oc.startswith('Бол'):
            return asian_total(M, L, over=True)
        if oc.startswith('Мен'):
            return asian_total(M, L, over=False)
        return None

    # ---------------- фора
    if mk in ('Фора', 'Азиатская фора'):
        m = re.match(r'Ф([12])\s*\(' + NUM + r'\)', oc)
        if not m:
            return None
        side = 'home' if m.group(1) == '1' else 'away'
        return asian_handicap(M, _f(m.group(2)), side)

    # ---------------- индивидуальные тоталы
    if mk in ('Индивидуальный тотал',) or low.startswith('инд. тотал'):
        m = re.match(r'ИТ([12])\s*\(' + NUM + r'\)', ln)
        if not m:
            return None
        side = 'home' if m.group(1) == '1' else 'away'
        L = _f(m.group(2))
        o, u, push = team_total(M, side, L)
        if oc.startswith('Бол'):
            return (o, push, u)
        if oc.startswith('Мен'):
            return (u, push, o)
        return None

    # ---------------- обе забьют
    if mk == 'Обе забьют':
        yes, no = btts(M)
        if oc == 'Да':
            return _pure(yes)
        if oc == 'Нет':
            return _pure(no)
        return None

    # ---------------- команда забьет
    if mk == 'Голы' and ln in ('К1', 'К2'):
        side = 'home' if ln == 'К1' else 'away'
        o, u, push = team_total(M, side, 0.5)
        if oc == 'Забьет':
            return _pure(o)
        if oc == 'Не забьет':
            return _pure(u)
        return None

    # ---------------- точный счёт
    if mk == 'Счет':
        m = re.match(r'^(\d+):(\d+)$', oc)
        if m:
            i, j = int(m.group(1)), int(m.group(2))
            if i < M.shape[0] and j < M.shape[0]:
                return _pure(float(M[i, j]))
        return None

    # ---------------- количество голов в матче
    if mk == 'Кол-во голов в матче':
        m = re.match(r'^(\d+)-(\d+)$', oc)
        if m:
            return _pure(interval_goals(M, int(m.group(1)), int(m.group(2))))
        m = re.match(r'^(\d+)\s*и более$', oc)
        if m:
            return _pure(interval_goals(M, int(m.group(1)), 99))
        return None

    # ---------------- точное количество голов команды
    if low.startswith('точное кол-во голов'):
        side = 'home' if home_name and home_name in mk else ('away' if away_name and away_name in mk else None)
        if side is None:
            return None
        x, y = _grid(M)
        t = x if side == 'home' else y
        m = re.match(r'^(\d+)$', oc)
        if m:
            return _pure(float(M[t == int(m.group(1))].sum()))
        m = re.match(r'^(\d+)\s*и более$', oc)
        if m:
            return _pure(float(M[t >= int(m.group(1))].sum()))
        return None

    # ---------------- чет/нечет индивидуального тотала
    if mk == 'Индивидуальный тотал голов' and ln in ('К1', 'К2'):
        x, y = _grid(M)
        t = x if ln == 'К1' else y
        odd = float(M[t % 2 == 1].sum())
        if oc == 'Нечет':
            return _pure(odd)
        if oc == 'Чет':
            return _pure(1 - odd)
        return None

    # ---------------- победит и тотал  /  не проиграет и тотал
    m = re.match(r'^П([12])\s*и\s*Т([БМ])\s*\(' + NUM + r'\)$', ln or mk)
    if m and oc in ('Да', 'Нет'):
        x, y = _grid(M)
        win = (x > y) if m.group(1) == '1' else (x < y)
        L = _f(m.group(3))
        tot = (x + y > L) if m.group(2) == 'Б' else (x + y < L)
        pr = float(M[win & tot].sum())
        return _pure(pr if oc == 'Да' else 1 - pr)

    # Х может быть кириллической (U+0425) -- у БЕТСИТИ именно так
    m = re.match(r'^(1[XХ]|[XХ]2|12)\s*и\s*Т([БМ])\s*\(' + NUM + r'\)$', ln or mk)
    if m and oc in ('Да', 'Нет'):
        x, y = _grid(M)
        tag = m.group(1).replace('Х', 'X')
        cond = (x >= y) if tag == '1X' else ((x <= y) if tag == 'X2' else (x != y))
        L = _f(m.group(3))
        tot = (x + y > L) if m.group(2) == 'Б' else (x + y < L)
        pr = float(M[cond & tot].sum())
        return _pure(pr if oc == 'Да' else 1 - pr)

    # ---------------- обе забьют и тотал
    if mk == 'Обе забьют и ТБ':
        m = re.search(NUM, ln)
        if m and oc in ('Да', 'Нет'):
            x, y = _grid(M)
            pr = float(M[(x > 0) & (y > 0) & (x + y > _f(m.group(1)))].sum())
            return _pure(pr if oc == 'Да' else 1 - pr)
        return None

    # ---------------- победит и не пропустит
    if mk == 'Победит и не пропустит' and ln in ('K1', 'К1', 'K2', 'К2'):
        x, y = _grid(M)
        cond = ((x > y) & (y == 0)) if ln in ('K1', 'К1') else ((y > x) & (x == 0))
        pr = float(M[cond].sum())
        if oc == 'Да':
            return _pure(pr)
        if oc == 'Нет':
            return _pure(1 - pr)
        return None

    # ---------------- победа ровно в N мячей
    if mk == 'Победа ровно в один мяч' or mk == 'Победа ровно в 1 мяч':
        return None

    return None


def evaluate(M, rows, home_name='', away_name=''):
    """Считает перевес по каждому исходу букмекера."""
    out = []
    for r in rows:
        res = price_bet(M, r['market'], r['line'], r['outcome'], home_name, away_name)
        if res is None:
            continue
        pw, pp, pl = res
        if pw <= 1e-6:
            continue
        price = r['price']
        fair = 1.0 + pl / pw                      # справедливый кэф с учётом возврата
        ev = pw * price - (1.0 - pp)              # матожидание на 1 ед. ставки
        out.append({**r, 'p_win': pw, 'p_push': pp, 'fair': fair,
                    'ev': ev, 'edge_pct': 100 * ev})
    return out
