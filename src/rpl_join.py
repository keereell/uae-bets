# -*- coding: utf-8 -*-
"""
Стыковка xG-данных РПЛ (365scores) с историческими коэффициентами
football-data.co.uk (RUS.csv: Pinnacle / максимум по рынку / среднее по рынку).

Сопоставление по дате (±1 день) и по нормализованным названиям команд.
"""
import os, re, sys
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# 365scores -> football-data
ALIAS = {
    'zenit': 'zenit', 'zenit st petersburg': 'zenit',
    'cska moscow': 'cska moscow', 'cska': 'cska moscow', 'pfc cska moscow': 'cska moscow',
    'spartak moscow': 'spartak moscow', 'spartak': 'spartak moscow',
    'lokomotiv moscow': 'lokomotiv moscow', 'lokomotiv': 'lokomotiv moscow',
    'dynamo moscow': 'dynamo moscow', 'dinamo moscow': 'dynamo moscow',
    'krasnodar': 'krasnodar', 'fc krasnodar': 'krasnodar',
    'rubin kazan': 'rubin kazan', 'rubin': 'rubin kazan',
    'akhmat grozny': 'akhmat grozny', 'akhmat': 'akhmat grozny',
    'rostov': 'rostov', 'fc rostov': 'rostov',
    'krylya sovetov': 'krylia sovetov', 'krylia sovetov': 'krylia sovetov',
    'krylya sovetov samara': 'krylia sovetov',
    'ural': 'ural', 'fc ural': 'ural',
    'orenburg': 'orenburg', 'fc orenburg': 'orenburg',
    'fakel': 'fakel voronezh', 'fakel voronezh': 'fakel voronezh',
    'pari nizhny novgorod': 'nizhny novgorod', 'nizhny novgorod': 'nizhny novgorod',
    'pari nn': 'nizhny novgorod',
    'sochi': 'sochi', 'fc sochi': 'sochi',
    'baltika': 'baltika', 'baltika kaliningrad': 'baltika',
    'akron': 'akron', 'akron togliatti': 'akron',
    'dynamo makhachkala': 'dinamo makhachkala', 'dinamo makhachkala': 'dinamo makhachkala',
    'khimki': 'khimki', 'fc khimki': 'khimki',
    'akhmat': 'akhmat grozny',
    'torpedo moscow': 'torpedo moscow',
    'akron tolyatti': 'akron', 'akron togliatti': 'akron',
    'ural yekaterinburg': 'ural',
    'rotor volgograd': 'rotor', 'rotor': 'rotor',
}


def norm(s):
    s = str(s).lower().strip()
    s = re.sub(r'\b(fc|fk|pfc|club)\b', '', s)
    s = re.sub(r'[^a-z ]', ' ', s)
    s = re.sub(r'\s+', ' ', s).strip()
    return ALIAS.get(s, s)


def main(fd_path):
    a = pd.read_csv(os.path.join(ROOT, 'data', 'rpl', 'matches.csv'))
    b = pd.read_csv(fd_path, encoding='utf-8-sig')
    b = b.dropna(subset=['MaxCH', 'MaxCD', 'MaxCA', 'Res'])
    b['d'] = pd.to_datetime(b.Date, format='%d/%m/%Y', errors='coerce')
    b = b.dropna(subset=['d'])
    b['h'] = b.Home.map(norm)
    b['a'] = b.Away.map(norm)

    a['d'] = pd.to_datetime(a.date, errors='coerce')
    a['h'] = a.home.map(norm)
    a['a'] = a.away.map(norm)

    # проверка соответствия названий
    na, nb = set(a.h) | set(a.a), set(b.h) | set(b.a)
    miss = sorted(x for x in na if x not in nb)
    if miss:
        print('НЕ СОПОСТАВЛЕНЫ (365scores -> football-data):', miss)

    merged = []
    bidx = {}
    for _, r in b.iterrows():
        bidx.setdefault((r.h, r.a), []).append(r)
    hit = 0
    for _, r in a.iterrows():
        cands = bidx.get((r.h, r.a), [])
        best = None
        for c in cands:
            if abs((c.d - r.d).days) <= 2:
                best = c
                break
        if best is None:
            continue
        hit += 1
        merged.append({**r.to_dict(),
                       'PSCH': best.PSCH, 'PSCD': best.PSCD, 'PSCA': best.PSCA,
                       'MaxCH': best.MaxCH, 'MaxCD': best.MaxCD, 'MaxCA': best.MaxCA,
                       'AvgCH': best.AvgCH, 'AvgCD': best.AvgCD, 'AvgCA': best.AvgCA,
                       'fd_res': best.Res})
    M = pd.DataFrame(merged)
    print(f'\nсостыковано матчей: {hit} из {len(a)} (365scores) / {len(b)} (football-data)')
    if len(M):
        print(f'с коэффициентами Pinnacle: {M.PSCH.notna().sum()}')
        print(f'с xG: {M.h_xg.notna().sum()}')
        # sanity: совпадают ли результаты
        r365 = np.where(M.hg > M.ag, 'H', np.where(M.hg == M.ag, 'D', 'A'))
        print(f'совпадение результатов между источниками: {(r365 == M.fd_res).mean():.1%}')
    out = os.path.join(ROOT, 'data', 'rpl', 'matches_odds.csv')
    M.to_csv(out, index=False, encoding='utf-8-sig')
    print('->', out)


if __name__ == '__main__':
    main(sys.argv[1])
