# -*- coding: utf-8 -*-
"""
ПРИЗНАКИ МАТЧА, которых нет в сырых данных.

Рейтинги атаки и обороны, подогнанные на xG, уже вобрали в себя всё, что
постоянно для команды: бюджет, качество состава, стиль. Смысл имеют только
величины, которые МЕНЯЮТСЯ ОТ МАТЧА К МАТЧУ внутри одной команды и сезона.
Здесь собраны ровно такие.

    python src/factors.py            # построить и сохранить data/factors.csv
    python src/factors.py --weather  # заодно догрузить погоду (13 запросов)

Погода берётся из архива Open-Meteo: бесплатно, без ключа, почасовая
температура и влажность по координатам стадиона. Один запрос на город
на весь диапазон дат, результат кэшируется в data/weather.csv.
"""
import os, sys, json, time, urllib.request, urllib.parse
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MATCHES = os.path.join(ROOT, 'data', 'matches.csv')
# 245 тысяч строк почасовых наблюдений: в сыром виде 8.5 МБ, в gzip 1.3 МБ
WEATHER = os.path.join(ROOT, 'data', 'weather.csv.gz')
OUT = os.path.join(ROOT, 'data', 'factors.csv')

# Координаты домашних городов. ОАЭ компактны: от Мадинат-Зайда до Диббы
# около 300 км по прямой, поэтому эффект перелётов заранее сомнителен --
# но проверить дешевле, чем спорить.
CITY = {
    'Al Ain':            (24.2075, 55.7447, 'Al Ain'),
    'Al Wasl':           (25.2048, 55.2708, 'Dubai'),
    'Al Nasr Dubai':     (25.2048, 55.2708, 'Dubai'),
    'Shabab Al Ahly':    (25.2048, 55.2708, 'Dubai'),
    'Dubai United':      (25.2048, 55.2708, 'Dubai'),
    'Al-Wahda':          (24.4539, 54.3773, 'Abu Dhabi'),
    'Jazira Abu Dhabi':  (24.4539, 54.3773, 'Abu Dhabi'),
    'Baniyas':           (24.3050, 54.6300, 'Baniyas'),
    'Al Dhafra':         (23.6500, 53.7000, 'Madinat Zayed'),
    'Sharjah SC':        (25.3463, 55.4209, 'Sharjah'),
    'Al Bataeh':         (25.2500, 55.7500, 'Al Bataeh'),
    'Ajman Club':        (25.4052, 55.5136, 'Ajman'),
    'Kalba':             (25.0500, 56.3500, 'Kalba'),
    'Khor Fakkan':       (25.3392, 56.3420, 'Khor Fakkan'),
    'Dibba Al Fujairah': (25.5942, 56.2694, 'Dibba'),
    'Dubba Al Husun':    (25.6197, 56.2739, 'Dibba Al-Hisn'),
    'Al Urooba':         (25.1288, 56.3265, 'Fujairah'),
    'Hatta Club':        (24.8000, 56.1167, 'Hatta'),
}

# Рамадан по григорианскому календарю. Даты сдвигаются примерно на 11 дней
# в год, поэтому таблица, а не формула.
RAMADAN = [
    ('2024-03-11', '2024-04-09'),
    ('2025-03-01', '2025-03-29'),
    ('2026-02-17', '2026-03-19'),
    ('2027-02-07', '2027-03-08'),
]

ARCHIVE = ('https://archive-api.open-meteo.com/v1/archive?latitude={lat}&longitude={lon}'
           '&start_date={d0}&end_date={d1}'
           '&hourly=temperature_2m,relative_humidity_2m&timezone=Asia%2FDubai')
# Архив отстаёт от сегодняшнего дня на несколько суток и на будущие даты
# отвечает 400. Ближайшие матчи закрываются прогнозом.
FORECAST = ('https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}'
            '&hourly=temperature_2m,relative_humidity_2m&timezone=Asia%2FDubai'
            '&past_days=7&forecast_days=16')


def haversine(a, b):
    """Расстояние между двумя точками по большому кругу, км."""
    R = 6371.0
    p1, p2 = np.radians(a[0]), np.radians(b[0])
    dp = p2 - p1
    dl = np.radians(b[1] - a[1])
    h = np.sin(dp / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2
    return float(2 * R * np.arcsin(np.sqrt(h)))


def _grab(url, city, rows, tries=4):
    """Два года почасовых данных -- ответ на 2 МБ, обрывы соединения обычны."""
    for k in range(tries):
        try:
            with urllib.request.urlopen(url, timeout=120) as r:
                j = json.loads(r.read().decode('utf-8'))
        except Exception as e:
            if k == tries - 1:
                print(f'  {city}: не удалось после {tries} попыток: {e}', file=sys.stderr)
                return 0
            time.sleep(2.0 * (k + 1))
            continue
        h = j.get('hourly') or {}
        for t, temp, rh in zip(h.get('time', []), h.get('temperature_2m', []),
                               h.get('relative_humidity_2m', [])):
            rows.append(dict(city=city, dt=t, temp=temp, rh=rh))
        return len(h.get('time', []))
    return 0


def fetch_weather(d0, d1, only=None):
    """
    Почасовая погода по каждому городу. Архив закрывает прошлое, прогноз --
    последнюю неделю и ближайшие две. Один запрос на город, не на матч.
    """
    seen, rows = {}, []
    for team, (lat, lon, city) in CITY.items():
        if city in seen or (only is not None and city not in only):
            continue
        seen[city] = True
        n = _grab(ARCHIVE.format(lat=lat, lon=lon, d0=d0, d1=d1), city, rows)
        n += _grab(FORECAST.format(lat=lat, lon=lon), city, rows)
        print(f'  {city}: {n} часов')
        time.sleep(0.3)
    if not rows:
        return pd.DataFrame(rows)
    return pd.DataFrame(rows).drop_duplicates(subset=['city', 'dt'], keep='last')


def load_weather(dates, refresh=False):
    """
    Догружает только недостающие города, а не всё заново: два года почасовых
    данных на город -- это мегабайты, и половина запросов обрывается.
    """
    have = pd.read_csv(WEATHER) if os.path.exists(WEATHER) else pd.DataFrame()
    want = {c for _, _, c in CITY.values()}
    # Архив обрывается за несколько суток до сегодня; на более поздние даты
    # он отвечает 400 и роняет весь запрос. Хвост добирает прогноз.
    cutoff = (pd.Timestamp.now('UTC').normalize() - pd.Timedelta(days=6)).strftime('%Y-%m-%d')
    d0, d1 = min(dates), min(max(dates), cutoff)
    if not refresh and not have.empty:
        return have
    # Городом считаем догруженным, только если у него есть архивная глубина,
    # а не одни лишь 550 часов прогноза.
    deep = set(have.groupby('city').size()[lambda s: s > 5000].index) if not have.empty else set()
    todo = sorted(want - deep)
    if not todo:
        return have
    print(f'гружу погоду: архив {d0}..{d1} + прогноз; городов осталось {len(todo)}')
    w = fetch_weather(d0, d1, only=set(todo))
    out = pd.concat([have, w], ignore_index=True) if not have.empty else w
    if not out.empty:
        out = out.drop_duplicates(subset=['city', 'dt'], keep='last')
        out['temp'] = out.temp.round(1)
        out['rh'] = out.rh.round(0)
        out.to_csv(WEATHER, index=False, compression='gzip')
    return out


def heat_index(t, rh):
    """
    Простой индекс жары (ощущаемая температура, °C). Формула Ротфуса
    в метрическом виде; ниже 27 °C влажность роли не играет.
    Именно связка «жарко + влажно» ограничивает работоспособность,
    а не температура сама по себе -- в ОАЭ влажность на побережье
    доходит до 90% при 30 °C, что тяжелее, чем сухие 38 °C в Аль-Айне.
    """
    t = np.asarray(t, float)
    rh = np.asarray(rh, float)
    hi = (-8.784695 + 1.61139411 * t + 2.338549 * rh - 0.14611605 * t * rh
          - 0.012308094 * t ** 2 - 0.016424828 * rh ** 2
          + 0.002211732 * t ** 2 * rh + 0.00072546 * t * rh ** 2
          - 0.000003582 * t ** 2 * rh ** 2)
    return np.where(t < 27.0, t, hi)


def in_ramadan(d):
    for a, b in RAMADAN:
        if a <= d <= b:
            return True
    return False


def build(refresh_weather=False):
    m = pd.read_csv(MATCHES)
    m = m.sort_values('ts').reset_index(drop=True)
    dates = sorted(m.date.astype(str).unique())

    # ---------- отдых и плотность календаря ----------
    # h_rest/a_rest в matches.csv уже есть, но их нет для первого матча
    # сезона и они не различают «отдыхал» и «не играл вообще».
    last = {}
    hist = {}
    rest_h, rest_a, cong_h, cong_a = [], [], [], []
    for _, r in m.iterrows():
        t = float(r.ts)
        for who, lst_r, lst_c in ((r.home, rest_h, cong_h), (r.away, rest_a, cong_a)):
            prev = last.get(who)
            lst_r.append(np.nan if prev is None else (t - prev) / 86400.0)
            past = hist.get(who, [])
            lst_c.append(sum(1 for x in past if 0 < (t - x) <= 21 * 86400))
        for who in (r.home, r.away):
            last[who] = t
            hist.setdefault(who, []).append(t)
    m['rest_h'], m['rest_a'] = rest_h, rest_a
    m['cong_h'], m['cong_a'] = cong_h, cong_a

    # Межсезонье -- это не отдых, а другое состояние. Больше 30 дней
    # обрезаем: разница между 40 и 90 днями простоя бессмысленна.
    for c in ('rest_h', 'rest_a'):
        m[c] = m[c].clip(upper=30.0)
    m['rest_diff'] = m.rest_h - m.rest_a          # асимметричный: + в пользу хозяев
    m['rest_min'] = m[['rest_h', 'rest_a']].min(axis=1)
    m['cong_sum'] = m.cong_h + m.cong_a           # симметричный: давит на обе команды
    m['cong_diff'] = m.cong_h - m.cong_a

    # ---------- переезды ----------
    def dist(row):
        a = CITY.get(row.home)
        b = CITY.get(row.away)
        if not a or not b:
            return np.nan
        return haversine(a[:2], b[:2])
    m['travel_a'] = m.apply(dist, axis=1)         # хозяева не едут никуда

    # ---------- время начала и календарь ----------
    m['evening'] = (m.kickoff_hour >= 19.0).astype(float)
    dd = pd.to_datetime(m.date)
    m['month'] = dd.dt.month
    m['ramadan'] = m.date.astype(str).map(in_ramadan).astype(float)

    # ---------- погода на момент начала ----------
    w = load_weather(dates, refresh=refresh_weather)
    if not w.empty:
        w['dt'] = pd.to_datetime(w.dt)
        w['date'] = w.dt.dt.strftime('%Y-%m-%d')
        w['hour'] = w.dt.dt.hour
        key = w.set_index(['city', 'date', 'hour'])[['temp', 'rh']]
        temps, rhs = [], []
        for _, r in m.iterrows():
            c = CITY.get(r.home)
            k = (c[2], str(r.date), int(round(float(r.kickoff_hour)))) if c else None
            if k is not None and k in key.index:
                v = key.loc[k]
                temps.append(float(v.temp)); rhs.append(float(v.rh))
            else:
                temps.append(np.nan); rhs.append(np.nan)
        m['temp'] = temps
        m['rh'] = rhs
        m['heat'] = heat_index(m.temp, m.rh)
    else:
        m['temp'] = m['rh'] = m['heat'] = np.nan

    cols = ['game_id', 'season', 'round', 'date', 'ts', 'home', 'away', 'played',
            'kickoff_hour', 'weekday', 'evening', 'month', 'ramadan',
            'rest_h', 'rest_a', 'rest_diff', 'rest_min',
            'cong_h', 'cong_a', 'cong_sum', 'cong_diff', 'travel_a',
            'temp', 'rh', 'heat']
    out = m[cols]
    out.to_csv(OUT, index=False)
    print(f'сохранено: {OUT}  строк {len(out)}')
    p = out[out.played.fillna(False)]
    print()
    print('заполненность на сыгранных (%d):' % len(p))
    for c in ('rest_diff', 'cong_sum', 'travel_a', 'temp', 'heat', 'ramadan'):
        print('  %-11s %3d  среднее %7.2f  разброс %6.2f  от %6.2f до %6.2f'
              % (c, p[c].notna().sum(), p[c].mean(), p[c].std(),
                 p[c].min(), p[c].max()))
    return out


if __name__ == '__main__':
    build(refresh_weather='--weather' in sys.argv)
