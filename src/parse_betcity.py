# -*- coding: utf-8 -*-
"""
Разбор сохранённых страниц БЕТСИТИ (.mhtml) в структурированную таблицу коэффициентов.

Структура страницы (Angular):
  div.line-event                      — шапка события
    .line-event__time                 — время начала
    .line-event__name-teams > b × 2   — команды
    .line-event__main-bets            — основная линия: 1 X 2, Ф1, Ф2, тотал
  div.dops-item                       — блок рынка
    .dops-item__title                 — название рынка
    .dops-item-row__section           — группа: [линия] + исходы
      .dops-item-row__block_single    — подпись линии (напр. «2.5» или «ИТ1 (1.5)»)
      .dops-item-row__block-left      — название исхода
      .dops-item-row__block-right     — коэффициент
"""
import time
import email, os, re, glob
from bs4 import BeautifulSoup

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MONTHS = {'января': 1, 'февраля': 2, 'марта': 3, 'апреля': 4, 'мая': 5, 'июня': 6,
          'июля': 7, 'августа': 8, 'сентября': 9, 'октября': 10, 'ноября': 11, 'декабря': 12}


def mhtml_to_html(path):
    with open(path, 'rb') as f:
        msg = email.message_from_binary_file(f)
    parts = [p.get_payload(decode=True) for p in msg.walk()
             if p.get_content_type() == 'text/html']
    return b'\n'.join(parts).decode('utf-8', errors='replace')


def _txt(el):
    return re.sub(r'\s+', ' ', el.get_text(' ', strip=True)).strip()


def _num(s):
    try:
        return float(str(s).replace(',', '.').strip())
    except (ValueError, TypeError):
        return None


def parse_header(ev):
    """Шапка: команды, время, основная линия."""
    out = {}
    t = ev.select_one('.line-event__time')
    if t:
        m = re.search(r'(\d{1,2}:\d{2})', _txt(t))
        if m:
            out['time'] = m.group(1)
    teams = [_txt(b) for b in ev.select('.line-event__name-teams b')]
    if len(teams) >= 2:
        out['home'], out['away'] = teams[0], teams[1]
    a = ev.select_one('a.line-event__name')
    if a and a.get('href'):
        out['url'] = a['href']

    # основная линия: последовательность кнопок/подписей
    seq = []
    for el in ev.select('.line-event__main-bets > *'):
        cls = el.get('class') or []
        txt = _txt(el)
        if 'line-event__main-bets-button_left' in cls or 'line-event__main-bets-button_no-value' in cls:
            seq.append(('label', txt))
        elif el.name == 'button':
            seq.append(('odd', _num(txt)))
    main = {}
    odds_only = [v for k, v in seq if k == 'odd' and v]
    if len(odds_only) >= 3:
        main['1'], main['X'], main['2'] = odds_only[0], odds_only[1], odds_only[2]
    labels = [v for k, v in seq if k == 'label']
    rest = [v for k, v in seq if k == 'odd'][3:]
    if len(labels) >= 3 and len(rest) >= 4:
        main['h1_line'], main['h1'] = _num(labels[0]), rest[0]
        main['h2_line'], main['h2'] = _num(labels[1]), rest[1]
        main['tot_line'], main['under'], main['over'] = _num(labels[2]), rest[2], rest[3]
    out['main'] = main
    return out


def parse_markets(soup):
    rows = []
    for item in soup.select('div.dops-item'):
        title_el = item.select_one('.dops-item__title')
        if title_el is None:
            continue
        spans = title_el.find_all('span', recursive=False)
        title = next((_txt(s) for s in spans if _txt(s) and 'icon' not in (s.get('class') or [])), '')
        for sec in item.select('.dops-item-row__section'):
            line_lbl = ''
            single = sec.select_one('.dops-item-row__block_single .dops-item-row__block-content')
            if single is not None:
                line_lbl = _txt(single)
            for blk in sec.select('.dops-item-row__block-content'):
                left = blk.select_one('.dops-item-row__block-left')
                right = blk.select_one('.dops-item-row__block-right')
                if left is None or right is None:
                    continue
                price = _num(_txt(right))
                if price is None:
                    continue
                rows.append(dict(market=title, line=line_lbl,
                                 outcome=_txt(left), price=price))
    return rows


def parse_page(path):
    soup = BeautifulSoup(mhtml_to_html(path), 'html.parser')
    for s in soup(['script', 'style', 'noscript']):
        s.decompose()
    ev = soup.select_one('div.line-event')
    head = parse_header(ev) if ev else {}

    # дата: из имени файла (дд.мм.гггг) или из текста страницы («4 сентября»)
    m = re.search(r'\((\d{2})\.(\d{2})\.(\d{4})\)', os.path.basename(path))
    if m:
        head['date'] = f'{m.group(3)}-{m.group(2)}-{m.group(1)}'
    else:
        txt = soup.get_text(' ')
        m2 = re.search(r'(\d{1,2})\s+(' + '|'.join(MONTHS) + r')', txt)
        if m2:
            head['date'] = f'2026-{MONTHS[m2.group(2)]:02d}-{int(m2.group(1)):02d}'
    head['file'] = os.path.basename(path)
    return head, parse_markets(soup)


def _from_api():
    """
    Линия БЕТСИТИ через их же публичный запрос, когда сохранённых страниц нет.
    Отдаёт только ОСНОВНЫЕ рынки (1X2, фора, тотал, двойной исход) — но именно
    на них и делаются ставки, а маржа на производных 13-58% всё равно
    исключает их из отбора.
    """
    import betcity_api
    out = []
    for g in betcity_api.snapshot(save=False):
        head = dict(home=g['home'], away=g['away'], date=(g['start'] or '')[:10],
                    time=(g['start'] or '')[11:16], file='api', main={})
        rows, main = [], {}
        for key, mk in g['markets'].items():
            kf, lv = mk['kf'], mk.get('line')
            if key.startswith('Фактический исход|'):
                o = {'P1': '1', 'X': 'X', 'P2': '2'}.get(key.split('|')[1])
                if o:
                    rows.append(dict(market='1X2', line='', outcome=o, price=kf))
                    main[o] = kf
            elif key.startswith('Двойной исход|'):
                rows.append(dict(market='Двойной исход', line='',
                                 outcome=key.split('|')[1], price=kf))
            elif key.startswith('Тотал|'):
                body = key.split('|')[1]
                oc = 'Мен' if body.startswith('Tm') else ('Бол' if body.startswith('Tb') else None)
                if oc is not None and lv is not None:
                    rows.append(dict(market='Тотал', line=f'{lv:g}', outcome=oc, price=kf))
                    main['tot_line'] = lv
                    main['under' if oc == 'Мен' else 'over'] = kf
            elif key.startswith('Фора|'):
                body = key.split('|')[1]
                tag = 'Ф1' if 'F1' in body else 'Ф2'
                if lv is not None:
                    rows.append(dict(market='Фора', line='',
                                     outcome=f"{tag} ({'+' if lv >= 0 else ''}{lv:g})", price=kf))
                    main[('h1_line' if tag == 'Ф1' else 'h2_line')] = lv
                    main['h1' if tag == 'Ф1' else 'h2'] = kf
        head['main'] = main
        out.append((head, rows))
    return out


FRESH_HOURS = 6.0


def load_all(folder=None, api_fallback=True, fresh_hours=FRESH_HOURS):
    """
    Свежие сохранённые страницы, а если их нет — линия через API.

    Сохранённая страница несёт ~340 рынков против 10 у API, поэтому она
    в приоритете. Но ТОЛЬКО свежая. Раньше любой файл в Bets/ молча
    перекрывал живую линию: 4 сентября 2026 все локальные расчёты весь день
    шли по страницам от 1 сентября (Шабаб П1 2.27 при живых 2.10), и два
    новых матча тура в них просто не существовали. CI тем временем ходил
    в API и видел правду. Расхождение вскрылось случайно.
    """
    folder = folder or os.path.join(ROOT, 'Bets')
    now = time.time()
    out, stale = [], []
    for p in sorted(glob.glob(os.path.join(folder, '*.mhtml'))):
        age_h = (now - os.path.getmtime(p)) / 3600.0
        if age_h > fresh_hours:
            stale.append((os.path.basename(p), age_h))
            continue
        head, rows = parse_page(p)
        out.append((head, rows))
    if stale:
        print(f'[линия] отброшено устаревших страниц: {len(stale)} (старше {fresh_hours:g} ч): '
              + ', '.join(f'{n} ({a/24:.1f} дн)' for n, a in stale[:6]))
    if not out and api_fallback:
        try:
            out = _from_api()
            print(f'[линия] свежих страниц нет, взял через API: {len(out)} матчей')
        except Exception as e:
            print(f'[линия] API недоступен: {e}')
    return out


if __name__ == '__main__':
    for head, rows in load_all():
        print('=' * 95)
        print(f"{head.get('home')} — {head.get('away')} | {head.get('date')} {head.get('time')}")
        print('  основная линия:', head.get('main'))
        print(f'  исходов разобрано: {len(rows)}')
        for key in ('Тотал', 'Фора', 'Обе забьют', 'Индивидуальный тотал', 'Двойной исход'):
            sel = [r for r in rows if r['market'] == key]
            if sel:
                print(f'  {key}: ' + '; '.join(
                    f"{r['line'] or ''} {r['outcome']}={r['price']}" for r in sel[:14]))
