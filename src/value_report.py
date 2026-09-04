# -*- coding: utf-8 -*-
"""
ОТЧЁТ ПО ТУРУ В ФОРМАТЕ «АНАЛИЗ — ЛИНИЯ — ЛУЧШАЯ СТАВКА».

Для каждого матча из папки Bets/:
  Анализ.  атака/оборона обеих команд в ожидаемых голах за матч (по xG)
  Линия.   что закладывает букмекер против того, что даёт модель
  Ставка.  лучший исход с перевесом в процентных пунктах + проверка
           устойчивости при смене метода снятия маржи («усадки»)

Ставки делятся на уровни:
  ОСНОВНАЯ   — перевес в коридоре, устойчив ко всем методам усадки, основной рынок
  СРЕДНЯЯ    — перевес в коридоре, но менее устойчив или рынок производный
  МЯГКАЯ     — та же идея на коротком кэфе (меньше дисперсия)
  СТАВКИ НЕТ — перевес ниже нижней границы коридора
  ОТБРОШЕНО  — перевес выше верхней границы (почти всегда ошибка модели)

    python src/value_report.py
"""
import sys, os, time
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model import DixonColes, detect_newcomers, score_matrix
from markets import wdl, DEVIG, margin, kelly
from backtest import walk_forward
from calibrate import fit_shrink, apply_shrink
from parse_betcity import load_all
from pricing import price_bet
from predict import best_params, MAIN_MARKETS
from teams import to_en
import pinnacle
from sharp import (pinnacle_constraints, fit_from_constraints, NAME_MAP,
                   fair_ev, shape_spread)
from implied import fit_implied
from round_calib import fit_round_calibration, apply_round_calibration, loo_calibration

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
pd.set_option('display.width', 250)

# Коридор задаётся в МАТОЖИДАНИИ, а не в процентных пунктах вероятности.
# Причина: EV = кэф * перевес_в_пп, то есть порог в п.п. означает совершенно
# разные требования на разных кэфах. Порог 3 п.п. при кэфе 1.35 требует EV +4%,
# а при кэфе 13.0 — уже +39%. Из-за этого матчи с аутсайдером-хозяином
# (Калба — Аль-Айн: вся линия на длинных кэфах) не давали НИ ОДНОГО кандидата,
# хотя модель расходилась с рынком по победе Калбы на 69% в относительном
# выражении (8.6% против 5.1%).
# Потолок задаётся ТОЛЬКО в п.п. перевеса (EDGE_HI ниже). Потолок по EV
# несовместим с полом по перевесу: EV = кэф * перевес, поэтому при перевесе
# 0.08 и кэфе выше 3.75 минимальный возможный EV уже больше 0.30 — ни один
# длинный кэф не проходил отбор НИКОГДА. Из-за этого пропадали кандидаты в
# матчах с явным аутсайдером-хозяином.
EV_LO, EV_HI = 0.04, float('inf')
# верхняя граница нужна по той же причине, что и раньше: слишком большой
# перевес почти всегда означает ошибку модели, а не находку
# Нижняя граница в п.п. -- это ШУМОВОЙ пол, а не пол доходности.
# Замер собственной ошибки модели: медиана бутстрап-SD вероятности исхода
# 5.5 п.п., медиана заявляемого перевеса 3.7 п.п. Доля исходов, где одна сигма
# модели БОЛЬШЕ заявленного перевеса: при 3 п.п. -- 39.6%, при 5 п.п. -- 25.4%,
# при 8 п.п. -- 6.5%, при 12 п.п. -- 0%. Берём 8 п.п.
# Итого ставка обязана пройти ОБА порога: перевес выше шума модели И
# матожидание выше порога доходности.
EDGE_LO, EDGE_HI = 0.08, 0.20
ODDS_LO, ODDS_HI = 1.35, 7.00
SOFT_ODDS = 1.75          # «мягкий вариант» — короткий кэф на ту же идею

# ГЛАВНЫЙ ФИЛЬТР. Ставка попадает в итог, только если она выгодна и против
# справедливой линии Pinnacle, а не только против нашей модели.
# Обоснование: та же стратегия без этого фильтра, прогнанная на 239 матчах
# с реальными коэффициентами, дала ROI -18.6% (78 ставок). Почти весь провал -
# это маржа: модель заявляла +18.2% EV, рынок закладывал -11.1%, факт -18.6%.
# Buchdahl на 24 150 ставках показал, что справедливые вероятности Pinnacle
# работают эталоном, а обратный тест корреляции не даёт.
# Острая линия НЕ вето, а мера уверенности. Причина: проверка блендинга дала
# вес модели рядом с рынком -0.21 с интервалом [-0.88, +0.41] по ОАЭ и +0.34
# с интервалом [-0.51, +0.99] по РПЛ. Ни один не значим. Данных не хватает,
# чтобы утверждать, что рынок всегда прав против модели, — значит, отбрасывать
# по нему нельзя, но и игнорировать глупо. Показываем как отдельный столбец.
PIN_MIN_EV = 0.01

# Калибровка сжатия возвращает k_s около нуля: предсказание ОБЩЕГО ТОТАЛА
# моделью неинформативно. Тоталы не убираем, но помечаем.
EXCLUDE_TOTALS = False
TOTALS_FAMILY = {'Тотал', 'Азиатский тотал', 'Индивидуальный тотал', 'Обе забьют',
                 'Обе забьют и ТБ', 'Победит и ТБ', 'Победит и ТМ',
                 'Кол-во голов в матче', 'Индивидуальный тотал голов'}


def team_rates(m, team):
    """Атака/оборона в ожидаемых голах за матч против среднего соперника."""
    i = m.idx[team]
    atk = float(np.exp(m.mu + m.atk[i]))
    dfn = float(np.exp(m.mu - m.dfn[i]))
    return atk, dfn


def devig_band(prices, idx):
    """Разброс подразумеваемой вероятности исхода по пяти методам усадки."""
    vals = []
    for nm in ('mult', 'add', 'power', 'shin', 'oddsratio'):
        try:
            vals.append(float(DEVIG[nm](list(prices))[idx]))
        except Exception:
            pass
    return (min(vals), max(vals)) if vals else (np.nan, np.nan)


def build():
    df = pd.read_csv(os.path.join(ROOT, 'data', 'matches.csv'))
    params = best_params()
    wf = walk_forward(df, params)
    cal = fit_shrink(wf.lh, wf.la, wf.hg, wf.ag)
    cal['k_s'] = float(np.clip(cal['k_s'], 0.0, 1.5))
    cal['k_d'] = float(np.clip(cal['k_d'], 0.3, 1.5))
    nc = detect_newcomers(df)
    up = df[~df.played].sort_values('ts')
    m = DixonColes(**params).fit(df, ref_ts=float(up.ts.min()), newcomers=nc[max(nc)])
    return df, m, cal, up, wf


def main():
    df, m, cal, up, wf = build()
    try:
        pin = pinnacle.parse()
    except Exception:
        pin = {}
    sharp, sharp_alt, pin_raw = {}, {}, {}
    for g in pin.values():
        key = (NAME_MAP.get(g['home'], g['home']), NAME_MAP.get(g['away'], g['away']))
        pin_raw[key] = g
        c = pinnacle_constraints(g)
        if len(c) >= 5:
            sharp[key] = fit_from_constraints(c)
            # Вторая форма под те же котировки. Она нужна не для точности,
            # а чтобы увидеть, насколько цена держится на форме, которую
            # Pinnacle не котирует. Расхождение двух форм -- честная
            # погрешность производного рынка.
            sharp_alt[key] = fit_from_constraints(c, fix_rho=0.0)

    # ---------- ПРОХОД 1: калибровка тура
    pages = load_all()

    # ОТСЕЧЬ СТАРТОВАВШИЕ. Список матчей приходит из ленты БЕТСИТИ, а она
    # какое-то время держит уже начавшиеся и даже сыгранные встречи.
    # 4 сентября 2026 отчёт спокойно посчитал Калбу — Аль-Айн (0:2) и
    # Аль-Васл — Хаур-Факкан (5:2) как предстоящие, с живой линией.
    # Ставку по ним предложить не успело только потому, что там не было
    # яруса 1. Сверяемся с расписанием, а не с доверием к ленте.
    now = time.time()
    fresh, dropped = [], []
    for head, brows in pages:
        h, a = to_en(head.get('home')), to_en(head.get('away'))
        row = df[(df.home == h) & (df.away == a)].sort_values('ts')
        row = row[row.ts >= now - 3 * 86400]
        started = row.empty or bool(row.iloc[0].played) or float(row.iloc[0].ts) <= now
        (dropped if started else fresh).append((head, brows))
    if dropped:
        print(f'Пропущено (матч уже начался или сыгран): '
              f'{", ".join(str(x[0].get("home")) + " — " + str(x[0].get("away")) for x in dropped)}')
    pages = fresh
    if not pages:
        print('\nНи одного предстоящего матча с линией.')
        return
    ml_pairs, mk_pairs, implied_cache = [], [], {}
    for head, brows in pages:
        h, a = to_en(head.get('home')), to_en(head.get('away'))
        if h not in m.idx or a not in m.idx:
            continue
        imp = fit_implied(brows, head.get('main', {}))
        if imp is None:
            continue
        lh0, la0 = m.lambdas(h, a)
        x, y = apply_shrink(np.array([lh0]), np.array([la0]),
                            cal['k_s'], cal['k_d'], cal['c'], cal['s_mean'])
        ml_pairs.append((float(x[0]), float(y[0])))
        mk_pairs.append((imp['lh'], imp['la']))
        implied_cache[(h, a)] = imp
    # история должна меряться на тех же СЖАТЫХ lambda, к которым потом
    # применяется поправка, иначе уровень корректируется дважды
    wf_sh = wf.copy()
    _lh, _la = apply_shrink(wf.lh.values, wf.la.values,
                            cal['k_s'], cal['k_d'], cal['c'], cal['s_mean'])
    wf_sh['lh'], wf_sh['la'] = _lh, _la
    dmu, dgam, rep = fit_round_calibration(ml_pairs, mk_pairs, wf=wf_sh)
    loo = loo_calibration(ml_pairs, mk_pairs, wf=wf_sh)
    loo_map = {}
    _i = 0
    for head_, brows_ in pages:
        h_, a_ = to_en(head_.get('home')), to_en(head_.get('away'))
        if h_ in m.idx and a_ in m.idx and (h_, a_) in implied_cache:
            loo_map[(h_, a_)] = loo[_i] if _i < len(loo) else (dmu, dgam)
            _i += 1

    picks = []
    print('#' * 104)
    print('ОТЧЁТ ПО ТУРУ — UAE PRO LEAGUE'.center(104))
    print('#' * 104)
    if rep:
        print("")
        t, hh = rep.get('тур'), rep.get('история')
        if t:
            print(f"Калибровка. По линии ({t['n']} матчей): модель давала тотал "
                  f"{t['тотал_модель']:.2f} против {t['тотал_рынок']:.2f} у рынка; "
                  f"поправки {t['dmu']:+.4f} / {t['dgamma']:+.4f} "
                  f"(ст.ошибки {t['se_mu']:.4f} / {t['se_gamma']:.4f}).")
        if hh:
            print(f"            По истории ({hh['n']} матчей вне выборки, только факт голов): "
                  f"поправки {hh['dmu']:+.4f} / {hh['dgamma']:+.4f} "
                  f"(ст.ошибки {hh['se_mu']:.4f} / {hh['se_gamma']:.4f}).")
        print(f"            Объединение по обратной дисперсии: уровень {dmu:+.4f}, "
              f"поле {dgam:+.4f}" + (" [обрезано по максимуму]" if rep.get('обрезано') else "") + ".")
        print("            Дальше показаны только ОТНОСИТЕЛЬНЫЕ расхождения с линией.")

    for head, brows in pages:
        h_ru, a_ru = head.get('home'), head.get('away')
        h, a = to_en(h_ru), to_en(a_ru)
        if h not in m.idx or a not in m.idx:
            print(f'\n!! {h_ru} — {a_ru}: нет в модели')
            continue

        lh0, la0 = m.lambdas(h, a)
        x, y = apply_shrink(np.array([lh0]), np.array([la0]),
                            cal['k_s'], cal['k_d'], cal['c'], cal['s_mean'])
        lh_raw, la_raw = float(x[0]), float(y[0])
        _dmu, _dgam = loo_map.get((h, a), (dmu, dgam))
        lh, la = apply_round_calibration(lh_raw, la_raw, _dmu, _dgam)
        M = score_matrix(lh, la, m.rho)
        p = wdl(M)
        ml = head.get('main', {})

        ha, hd = team_rates(m, h)
        aa, ad = team_rates(m, a)

        print('\n' + '─' * 104)
        print(f'{h_ru} — {a_ru}  ({head.get("date")}, {head.get("time")})')
        print('─' * 104)
        print(f'Анализ. {h_ru} {ha:.2f}/{hd:.2f}, {a_ru} {aa:.2f}/{ad:.2f} '
              f'(атака/оборона, ожидаемые голы за матч по xG). '
              f'Модель ждёт счёт {lh:.2f}:{la:.2f}, тотал {lh+la:.2f} '
              f'(до калибровки тура было {lh_raw:.2f}:{la_raw:.2f}; '
              f'поправки без участия этого матча {_dmu:+.4f}/{_dgam:+.4f}).')

        if all(k in ml for k in ('1', 'X', '2')):
            o = [ml['1'], ml['X'], ml['2']]
            q = DEVIG['power'](o)
            print(f'Линия. {o[0]:.2f} / {o[1]:.2f} / {o[2]:.2f}, маржа {100*margin(o):.1f}%. '
                  f'Рынок даёт хозяевам {100*q[0]:.1f}%, модель {100*p["H"]:.1f}%.')

        # ---- собираем все исходы
        mr = list(brows)
        if all(k in ml for k in ('1', 'X', '2')):
            for nm in ('1', 'X', '2'):
                mr.append(dict(market='1X2', line='', outcome=nm, price=ml[nm]))
        if ml.get('tot_line') is not None:
            mr.append(dict(market='Тотал', line=f"{ml['tot_line']:g}", outcome='Мен', price=ml['under']))
            mr.append(dict(market='Тотал', line=f"{ml['tot_line']:g}", outcome='Бол', price=ml['over']))
        for kl, ko, tag in (('h1_line', 'h1', 'Ф1'), ('h2_line', 'h2', 'Ф2')):
            if ml.get(kl) is not None:
                v = ml[kl]
                mr.append(dict(market='Фора', line='',
                               outcome=f"{tag} ({'+' if v >= 0 else ''}{v:g})", price=ml[ko]))

        Ms = sharp.get((h, a), {}).get('M')
        Ms_alt = sharp_alt.get((h, a), {}).get('M')
        cand = []
        for r in mr:
            pm = price_bet(M, r['market'], r['line'], r['outcome'], h_ru, a_ru)
            if pm is None or pm[0] <= 1e-6:
                continue
            w, pu, l = pm
            price = r['price']
            if not (ODDS_LO <= price <= ODDS_HI):
                continue
            fair = 1.0 + l / w
            implied = (1.0 - pu) / price          # сырая вероятность из кэфа (с маржой внутри)
            edge_pp = w - implied                 # перевес в п.п., как считают в роликах
            ev = w * price - (1.0 - pu)           # честное матожидание на 1 ед. ставки
            p_cond = w / max(w + l, 1e-9)         # вероятность выигрыша при условии «не возврат»
            # устойчивость к усадке: сравниваем с честной ценой по двусторонней паре,
            # приближая её разбросом методов на 1X2, если рынок трёхсторонний
            band = None
            if r['market'] == '1X2' and all(k in ml for k in ('1', 'X', '2')):
                i = {'1': 0, 'X': 1, '2': 2}[r['outcome']]
                lo, hi = devig_band([ml['1'], ml['X'], ml['2']], i)
                band = (w - hi, w - lo)           # худший и лучший перевес
            # Прямая цена Pinnacle там, где он торгует рынок сам; подгонка —
            # только для остальных. Подгонка добавляет собственную ошибку формы:
            # на 1X2 Шабаба она давала -3.2% против -3.7...-5.0% у прямого де-вига.
            ppin, pin_src, p_sharp = fair_ev(pin_raw.get((h, a)), Ms, r['market'],
                                             r['line'], r['outcome'], price, h_ru, a_ru)
            # Для производных рынков берём ХУДШУЮ из двух форм. Оптимистичная
            # оценка на неподкреплённом параметре -- это не находка, а иллюзия.
            pin_alt = (shape_spread(Ms_alt, r['market'], r['line'], r['outcome'],
                                    price, h_ru, a_ru)
                       if pin_src == 'подгонка' else None)
            if ppin is not None and pin_alt is not None:
                ppin = min(ppin, pin_alt)
            cand.append(dict(матч=f'{h_ru} — {a_ru}', рынок=r['market'], линия=r['line'],
                             исход=r['outcome'], кэф=price, p=w, p_cond=p_cond, fair=fair,
                             edge_pp=edge_pp, ev=ev, band=band, ev_pin=ppin,
                             pin_src=pin_src, p_sharp=p_sharp, ev_pin_alt=pin_alt,
                             осн=r['market'] in MAIN_MARKETS))

        C = pd.DataFrame(cand)
        if C.empty:
            print('Ставок нет: не удалось оценить ни один исход.')
            continue

        C['прошёл_pin'] = C.ev_pin.notna() & (C.ev_pin > PIN_MIN_EV)
        C['тотальный'] = C['рынок'].isin(TOTALS_FAMILY) | C['рынок'].str.startswith('Точное кол-во')
        in_corridor = ((C.ev >= EV_LO) & (C.ev <= EV_HI)
                       & (C.edge_pp >= EDGE_LO) & (C.edge_pp <= EDGE_HI))
        # шум подгонки линии Pinnacle именно для этого матча
        # Полоса «рынок молчит» — это погрешность ПОДГОНКИ. У прямой цены
        # Pinnacle её нет, поэтому там полоса узкая: только разброс методов
        # снятия маржи, около 1 п.п.
        noise_fit = 2.0 * float(sharp.get((h, a), {}).get('rmse', 0.02))
        noise_direct = 0.01
        C['шум'] = np.where(C.pin_src == 'прямая', noise_direct, noise_fit)
        C['ярус'] = np.where(
            ~in_corridor, '',
            np.where(C.ev_pin.isna(), '?',
                     np.where(C.ev_pin > C['шум'], '1',
                              np.where(C.ev_pin >= -C['шум'], '2', '3'))))
        tier1 = C[C['ярус'] == '1'].sort_values('ev', ascending=False)
        tier2 = C[C['ярус'] == '2'].sort_values('ev', ascending=False)
        tier3 = C[C['ярус'] == '3'].sort_values('ev', ascending=False)
        tierQ = C[C['ярус'] == '?'].sort_values('ev', ascending=False)
        above = C[(C.ev > EV_HI) | (C.edge_pp > EDGE_HI)].sort_values('ev', ascending=False)
        n_corr = int(in_corridor.sum())

        print(f'Отбор. В коридоре EV {EV_LO:.0%}-{EV_HI:.0%}: {n_corr} исходов. '
              f'Рынок подтверждает: {len(tier1)}; молчит '
              f'(±{100*noise_direct:.1f}% на прямой цене, ±{100*noise_fit:.1f}% на подгонке): '
              f'{len(tier2)}; возражает: {len(tier3)}; не торгует: {len(tierQ)}.')

        def _line(r, tag):
            band_txt = ''
            if r['band'] is not None and not any(np.isnan(v) for v in r['band']):
                band_txt = (f', против честной линии {100*r["band"][0]:+.1f}…'
                            f'{100*r["band"][1]:+.1f} п.п.')
            if r['ev_pin'] is None or (isinstance(r['ev_pin'], float) and np.isnan(r['ev_pin'])):
                pin_txt = ' | Pinnacle не торгует этот исход'
            elif r['ev_pin'] > 0:
                pin_txt = f' | против Pinnacle {100*r["ev_pin"]:+.1f}%'
            else:
                pin_txt = f' | НЕДОБОР до справедливой цены {100*abs(r["ev_pin"]):.1f}%'
            print(f'{tag}: {r["рынок"]} {r["линия"]} {r["исход"]} @{r["кэф"]:.2f} — '
                  f'модель {100*r["p"]:.1f}%, справедливый {r["fair"]:.2f}, '
                  f'перевес {100*r["edge_pp"]:+.1f} п.п.{band_txt}{pin_txt}')

        for tf, tag in ((tier1, 'ПОДТВЕРЖДЕНО'), (tier2, 'РЫНОК МОЛЧИТ'), (tierQ, 'НЕТ ЭТАЛОНА')):
            if len(tf):
                _line(tf.iloc[0], tag)
                picks.append(tf.iloc[0])
        tierB = tier3
        if len(tierB):
            # берём до трёх разных рынков на матч, чтобы не дублировать одну идею
            seen, taken = set(), 0
            for _, r in tierB.iterrows():
                key = (r['рынок'], r['исход'][:2])
                if key in seen:
                    continue
                seen.add(key)
                _line(r, 'РЫНОК ПРОТИВ' if taken == 0 else '            ')
                picks.append(r)
                taken += 1
                if taken >= 3:
                    break
        if not n_corr:
            b = C.sort_values('edge_pp', ascending=False).iloc[0]
            print(f'Ставки нет ни в одном ярусе. Лучший исход — {b["исход"]} '
                  f'@{b["кэф"]:.2f}, перевес {100*b["edge_pp"]:+.1f} п.п.')

        if len(above):
            names = ', '.join(f'{r["исход"]} @{r["кэф"]:.2f} (EV {100*r["ev"]:+.0f}%)'
                              for _, r in above.head(3).iterrows())
            print(f'Отброшено как слишком жирное (EV >{EV_HI:.0%}): {names} — всего {len(above)}.')

    # ---------------- сводка тура
    print('\n' + '#' * 104)
    print('ИТОГ ТУРА'.center(104))
    print('#' * 104)
    if not picks:
        print('\nНи одной ставки в коридоре.')
        return
    P = pd.DataFrame(picks).reset_index(drop=True)
    P = P.drop_duplicates(subset=['матч', 'рынок', 'линия', 'исход'])
    # РАЗМЕР СЧИТАЕТСЯ ПО ОСТРОЙ ЛИНИИ, А НЕ ПО МОДЕЛИ.
    # Проверено на 717 исходах walk-forward: в КАЖДОМ бакете расхождения
    # модель-рынок факт оказывался ближе к рынку. На краях модель ошибается
    # на 14 п.п. (говорит 50.9%, рынок 36.6%, факт 36.4%), рынок -- на 0.2 п.п.
    # Бриер 0.1902 против 0.1846. Ставки по перевесу модели: ROI -23.1%
    # на 207 исходах. Поэтому в Келли идёт вероятность, выведенная из Pinnacle;
    # модель остаётся фильтром «на что вообще смотреть», а не мерой уверенности.
    # Если эталона нет, ставка не играется, и размер не нужен.
    P['Келли_%'] = [100 * kelly(r.p_sharp, r.кэф, frac=0.25, cap=0.02)
                    if pd.notna(r.p_sharp) else 0.0
                    for r in P.itertuples()]

    T1 = P[P['ярус'] == '1'].sort_values('ev', ascending=False)
    T2 = P[P['ярус'] == '2'].sort_values('ev', ascending=False)
    T3 = P[P['ярус'] == '3'].sort_values('ev', ascending=False)
    TQ = P[P['ярус'] == '?'].sort_values('ev', ascending=False)

    def block(title, note, d, show_kelly=False):
        print('')
        print(title)
        for n in note:
            print('   ' + n)
        if d.empty:
            print('   пусто')
            return
        for _, r in d.iterrows():
            if pd.isna(r['ev_pin']):
                pin = 'эталона нет'
            else:
                pin = f'рынок {100*r["ev_pin"]:+.1f}%'
            flag = ' [тотал]' if r['тотальный'] else ''
            k = f' | Келли {r["Келли_%"]:.2f}%' if show_kelly else ''
            print(f'   {r["матч"]}, {r["рынок"]} {r["линия"]} {r["исход"]} @{r["кэф"]:.2f} | '
                  f'модель {100*r["p"]:.1f}% | справ. {r["fair"]:.2f} | '
                  f'перевес {100*r["edge_pp"]:+.1f} п.п. | EV {100*r["ev"]:+.1f}% | '
                  f'{pin}{flag}{k}')

    block('1. ПОДТВЕРЖДЕНО ОСТРОЙ ЛИНИЕЙ',
          ['перевес по xG-модели, и цена выше справедливой по Pinnacle',
           'единственный отбор с проверенной доходностью (Buchdahl, 24 150 ставок: +1.81%)'],
          T1, show_kelly=True)

    block('2. РЫНОК НЕ ВОЗРАЖАЕТ  (НЕ ИГРАЕТСЯ)',
          ['перевес по xG-модели, расхождение с Pinnacle в пределах погрешности',
           'острая линия не подтверждает, но и не опровергает — а одного перевеса',
           'модели недостаточно: на 207 исходах такой отбор дал ROI -23.1%'],
          T2)

    block('3. РЫНОК ВОЗРАЖАЕТ',
          ['перевес по xG-модели, но цена заметно ниже справедливой по Pinnacle',
           'ровно такой отбор на 239 матчах дал ROI -18.6% (78 ставок)'],
          T3)

    block('4. ЭТАЛОНА НЕТ',
          ['Pinnacle этот рынок не торгует, проверить нечем'],
          TQ)

    # Играется ТОЛЬКО ярус 1. Он единственный не опирается на правоту модели:
    # там цена выше справедливой по острой линии, то есть перевес берётся
    # из сравнения двух цен, а не из прогноза. Ровно такие стратегии и
    # показывают прибыль в литературе (Kaunitz и др. 2017: правило
    # «максимальный кэф выше консенсуса», модель не используется вовсе;
    # Buchdahl: 24 150 ставок, +1.81%). Ярус 2 держится на перевесе модели,
    # а он на этих данных отрицательно доходен.
    play = T1
    if not play.empty:
        tot = float(play['Келли_%'].sum())
        scale = min(1.0, 6.0 / tot) if tot > 0 else 1.0
        print('')
        print(f'Размер ставок (только ярус 1, четверть-Келли по вероятности '
              f'острой линии, потолок 6% банка на тур): {tot*scale:.1f}%')
        for _, r in play.iterrows():
            print(f'   {r["матч"]}, {r["исход"]} @{r["кэф"]:.2f} -> '
                  f'{r["Келли_%"]*scale:.2f}% банка')

    P.drop(columns=['band']).to_csv(os.path.join(ROOT, 'data', 'picks.csv'),
                                    index=False, encoding='utf-8-sig')
    print('Сохранено: data/picks.csv')


if __name__ == '__main__':
    main()
