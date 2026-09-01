# -*- coding: utf-8 -*-
"""Разведочный анализ: базовые ставки лиги, преимущество поля, время начала, качество xG."""
import os, numpy as np, pandas as pd
pd.set_option('display.width', 200); pd.set_option('display.max_columns', 50)
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
df = pd.read_csv(os.path.join(ROOT, 'data', 'matches.csv'))
p = df[df.played].copy()
p['res'] = np.where(p.hg > p.ag, 'H', np.where(p.hg == p.ag, 'D', 'A'))

print('=== 1. БАЗОВЫЕ ПОКАЗАТЕЛИ ЛИГИ ===')
for s, g in p.groupby('season'):
    print(f"сезон {s}: N={len(g):3d}  голы {g.hg.mean():.2f}-{g.ag.mean():.2f} "
          f"(всего {(g.hg+g.ag).mean():.2f})  H/D/A = "
          f"{(g.res=='H').mean():.1%}/{(g.res=='D').mean():.1%}/{(g.res=='A').mean():.1%}")
print(f"ВСЕГО:     N={len(p):3d}  голы {p.hg.mean():.2f}-{p.ag.mean():.2f} "
      f"(всего {(p.hg+p.ag).mean():.2f})  H/D/A = "
      f"{(p.res=='H').mean():.1%}/{(p.res=='D').mean():.1%}/{(p.res=='A').mean():.1%}")

x = p.dropna(subset=['h_xg', 'a_xg'])
print(f"\nxG (N={len(x)}): дома {x.h_xg.mean():.2f}  в гостях {x.a_xg.mean():.2f}  "
      f"итого {(x.h_xg+x.a_xg).mean():.2f}")
print(f"преимущество поля: голы +{p.hg.mean()-p.ag.mean():.3f}  xG +{x.h_xg.mean()-x.a_xg.mean():.3f}  "
      f"мультипликатор xG {x.h_xg.mean()/x.a_xg.mean():.3f}")
print(f"реализация: голы/xG дома {x.hg.sum()/x.h_xg.sum():.3f}, в гостях {x.ag.sum()/x.a_xg.sum():.3f}")

print('\n=== 2. ВРЕМЯ НАЧАЛА МАТЧА ===')
p['slot'] = pd.cut(p.kickoff_hour, [0, 17.0, 18.5, 21.0, 24], labels=['<17', '17-18:30', '18:30-21', '>21'])
agg = p.groupby('slot', observed=True).agg(n=('hg','size'), gf=('hg','mean'), ga=('ag','mean'),
                                           tot=('hg', lambda s: 0), hxg=('h_xg','mean'), axg=('a_xg','mean'))
agg['total_goals'] = p.groupby('slot', observed=True).apply(lambda g: (g.hg+g.ag).mean(), include_groups=False)
agg['home_win'] = p.groupby('slot', observed=True).apply(lambda g: (g.res=='H').mean(), include_groups=False)
agg['draw'] = p.groupby('slot', observed=True).apply(lambda g: (g.res=='D').mean(), include_groups=False)
print(agg.drop(columns=['tot']).round(3))

print('\n=== 3. МЕСЯЦ (жара) ===')
p['month'] = pd.to_datetime(p.date).dt.month
m = p.groupby('month').apply(lambda g: pd.Series({
    'n': len(g), 'goals': (g.hg+g.ag).mean(), 'xg': (g.h_xg+g.a_xg).mean(),
    'home_win': (g.res=='H').mean(), 'shots': (g.h_shots+g.a_shots).mean()}), include_groups=False)
print(m.round(3))

print('\n=== 4. ОТДЫХ ===')
p['rest_diff'] = p.h_rest - p.a_rest
q = p.dropna(subset=['rest_diff'])
q = q.assign(bucket=pd.cut(q.rest_diff, [-99, -3, -1, 1, 3, 99]))
print(q.groupby('bucket', observed=True).apply(lambda g: pd.Series({
    'n': len(g), 'gd': (g.hg-g.ag).mean(), 'home_win': (g.res=='H').mean()}), include_groups=False).round(3))

print('\n=== 5. КОЭФФИЦИЕНТЫ Bet365: маржа и калибровка ===')
o = p.dropna(subset=['odds_H','odds_D','odds_A']).copy()
inv = 1/o.odds_H + 1/o.odds_D + 1/o.odds_A
print(f"N={len(o)}  средний overround = {inv.mean():.4f}  (маржа {(inv.mean()-1)*100:.2f}%)")
for c, lab in (('H','дом'), ('D','ничья'), ('A','гости')):
    pr = (1/o['odds_'+c])/inv
    act = (o.res == c).mean()
    print(f"  {lab}: импл.вер. {pr.mean():.3f}  факт {act:.3f}")
# favourite-longshot
o['pH'] = (1/o.odds_H)/inv
bins = pd.cut(o.pH, [0,.2,.35,.5,.65,.8,1])
print(o.groupby(bins, observed=True).apply(lambda g: pd.Series({
    'n': len(g), 'pred': g.pH.mean(), 'act': (g.res=='H').mean()}), include_groups=False).round(3))

print('\n=== 6. ПРЕДСКАЗАТЕЛЬНАЯ СИЛА xG vs ГОЛЫ (скользящее окно) ===')
# для каждой команды считаем скользящее среднее за последние K матчей и смотрим корреляцию со следующим матчем
long = []
for _, r in p.sort_values('ts').iterrows():
    long.append(dict(ts=r.ts, team=r.home_id, opp=r.away_id, gf=r.hg, ga=r.ag,
                     xgf=r.h_xg, xga=r.a_xg, home=1))
    long.append(dict(ts=r.ts, team=r.away_id, opp=r.home_id, gf=r.ag, ga=r.hg,
                     xgf=r.a_xg, xga=r.h_xg, home=0))
L = pd.DataFrame(long).sort_values('ts').reset_index(drop=True)
for K in (5, 8, 12, 20):
    rows = []
    for t, g in L.groupby('team'):
        g = g.sort_values('ts')
        for col in ('gf','ga','xgf','xga'):
            g['r_'+col] = g[col].shift(1).rolling(K, min_periods=K).mean()
        rows.append(g)
    R = pd.concat(rows).dropna(subset=['r_gf','r_xgf','gf','xgf'])
    c = lambda a,b: np.corrcoef(R[a], R[b])[0,1]
    print(f"K={K:2d} N={len(R):4d} | предсказание СЛЕД. ГОЛОВ: по прошлым голам r={c('r_gf','gf'):.3f}, "
          f"по прошлым xG r={c('r_xgf','gf'):.3f} || след. xG: голы r={c('r_gf','xgf'):.3f}, xG r={c('r_xgf','xgf'):.3f}")
    print(f"        пропущено: по прошлым GA r={c('r_ga','ga'):.3f}, по прошлым xGA r={c('r_xga','ga'):.3f}")
