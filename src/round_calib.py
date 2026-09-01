# -*- coding: utf-8 -*-
"""
КАЛИБРОВКА ТУРА, устойчивая к малому числу матчей.

Задача. Модель может быть смещена относительно линии целиком — например,
давать в среднем меньше голов. Тогда «перевес» появится в каждом матче тура,
и это будет не несогласие с рынком по конкретной паре, а одна системная
ошибка, размноженная на весь тур.

Поправки:
    log λ_дом'  = log λ_дом  + dμ + dγ
    log λ_гост' = log λ_гост + dμ

Проблема. Если в туре открыто мало матчей (у нас четыре), среднее по ним
шумное, и калибровка начнёт съедать настоящий сигнал.

Решение. Две независимые оценки одного и того же сдвига, объединённые
по обратной дисперсии:

  1) ПО ТУРУ — сравнение λ модели с λ, восстановленными из линии букмекера.
     Точная, но по малому числу матчей; её разброс меряем прямо по матчам.

  2) ПО ИСТОРИИ — сравнение прогнозов walk-forward с ФАКТИЧЕСКИМИ голами.
     Коэффициенты не нужны вообще, выборка в десятки раз больше, но
     оценка шумная из-за случайности самих голов.

     dμ_ист  = log( сумма фактических голов гостей / сумма λ гостей )
     dγ_ист  = log( сумма фактических голов хозяев / сумма λ хозяев ) − dμ_ист

Итог: dμ = (dμ_тур/σ²_тур + dμ_ист/σ²_ист) / (1/σ²_тур + 1/σ²_ист)
"""
import numpy as np


def _mean_se(v):
    v = np.asarray(v, float)
    n = len(v)
    if n == 0:
        return 0.0, np.inf
    if n == 1:
        return float(v[0]), 0.25          # одиночное наблюдение: широкая неопределённость
    return float(v.mean()), float(v.std(ddof=1) / np.sqrt(n))


def round_estimate(model_lams, market_lams):
    """Сдвиг по туру: сравнение с восстановленными из линии λ."""
    mh = np.array([x[0] for x in model_lams], float)
    ma = np.array([x[1] for x in model_lams], float)
    kh = np.array([x[0] for x in market_lams], float)
    ka = np.array([x[1] for x in market_lams], float)
    ok = (mh > 0) & (ma > 0) & (kh > 0) & (ka > 0)
    if ok.sum() == 0:
        return None
    d_away = np.log(ka[ok]) - np.log(ma[ok])
    d_home = np.log(kh[ok]) - np.log(mh[ok])
    dmu, se_mu = _mean_se(d_away)
    dh, _ = _mean_se(d_home)
    # d_home и d_away сильно коррелированы (на текущем туре -0.64), поэтому
    # ст.ошибку разности считаем ПАРНО, а не через hypot независимых оценок
    _, se_gam = _mean_se(d_home - d_away)
    return dict(n=int(ok.sum()), dmu=dmu, se_mu=se_mu,
                dgamma=dh - dmu, se_gamma=se_gam,
                тотал_модель=float((mh + ma)[ok].mean()),
                тотал_рынок=float((kh + ka)[ok].mean()))


def history_estimate(wf):
    """
    Сдвиг по истории: walk-forward прогнозы против фактических голов.
    wf: DataFrame с колонками lh, la, hg, ag.
    """
    lh = wf.lh.values.astype(float)
    la = wf.la.values.astype(float)
    hg = wf.hg.values.astype(float)
    ag = wf.ag.values.astype(float)
    n = len(wf)
    if n < 30:
        return None
    dmu = float(np.log(ag.sum() / la.sum()))
    dh = float(np.log(hg.sum() / lh.sum()))
    # дисперсия: голы пуассоновские, дисперсия log-отношения ~ 1/сумма голов
    se_mu = float(np.sqrt(1.0 / max(ag.sum(), 1.0)))
    se_h = float(np.sqrt(1.0 / max(hg.sum(), 1.0)))
    return dict(n=n, dmu=dmu, se_mu=se_mu,
                dgamma=dh - dmu, se_gamma=float(np.hypot(se_h, se_mu)))


def _combine(a, sa, b, sb):
    wa = 1.0 / max(sa, 1e-6) ** 2
    wb = 1.0 / max(sb, 1e-6) ** 2
    return (a * wa + b * wb) / (wa + wb), float(np.sqrt(1.0 / (wa + wb)))


def loo_calibration(model_lams, market_lams, wf=None, max_shift=0.25):
    """
    Поправки, посчитанные БЕЗ участия каждого матча в его собственной калибровке.
    Иначе матч подтягивает сам себя к линии и часть его расхождения исчезает
    ещё до подсчёта перевеса (при четырёх матчах — около 10% расхождения).
    -> список (dmu_i, dgamma_i) по матчам в исходном порядке
    """
    n = len(model_lams)
    out = []
    for i in range(n):
        ml = [x for j, x in enumerate(model_lams) if j != i]
        mk = [x for j, x in enumerate(market_lams) if j != i]
        if len(ml) >= 2:
            dmu, dgam, _ = fit_round_calibration(ml, mk, wf=wf, max_shift=max_shift)
        else:
            dmu, dgam, _ = fit_round_calibration([], [], wf=wf, max_shift=max_shift)
        out.append((dmu, dgam))
    return out


def fit_round_calibration(model_lams, market_lams, wf=None, max_shift=0.25):
    """
    -> (dmu, dgamma, отчёт)
    max_shift ограничивает поправку: калибровка должна снимать смещение,
    а не переписывать модель.
    """
    r = round_estimate(model_lams, market_lams)
    h = history_estimate(wf) if wf is not None else None
    if r is None and h is None:
        return 0.0, 0.0, {}
    if r is None:
        dmu, se_mu = h['dmu'], h['se_mu']
        dgam, se_g = h['dgamma'], h['se_gamma']
    elif h is None:
        dmu, se_mu = r['dmu'], r['se_mu']
        dgam, se_g = r['dgamma'], r['se_gamma']
    else:
        dmu, se_mu = _combine(r['dmu'], r['se_mu'], h['dmu'], h['se_mu'])
        dgam, se_g = _combine(r['dgamma'], r['se_gamma'], h['dgamma'], h['se_gamma'])

    dmu_c = float(np.clip(dmu, -max_shift, max_shift))
    dgam_c = float(np.clip(dgam, -max_shift, max_shift))
    rep = dict(тур=r, история=h, dmu=dmu_c, dgamma=dgam_c,
               se_mu=se_mu, se_gamma=se_g,
               обрезано=(abs(dmu - dmu_c) > 1e-9 or abs(dgam - dgam_c) > 1e-9))
    return dmu_c, dgam_c, rep


def apply_round_calibration(lh, la, dmu, dgamma):
    return float(np.exp(np.log(lh) + dmu + dgamma)), float(np.exp(np.log(la) + dmu))
