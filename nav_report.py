#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""nav_report — сводная фигура «курс / угол на цель / ошибка курса» по
nav_session2.csv. Строится автоматически в конце start_stz и отдельным
запуском: <OUT_DIR>/nav_yaw_vs_bearing.jpg.

Панели (3x2):
  а) курс от угла на цель      б) устойчивость курса в бинах угла
  в) распределение курса       г) ход курса по времени (absolute:
                                  развёрнутая траектория + сглаживание)
  д) гистограмма ошибки курса   е) ошибка курса по кадрам
Ошибка курса — относительно опоры yaw_true = A0 - beta; якорь A0
берётся из чекпойнта дообучения (settings.NAV_CKPT -> finetune.
anchors_deg), иначе — круговая медиана (yaw + beta) по CSV. Акцент —
требование «не хуже ±5°»: полоса ±5°, медиана с бутстреп-ДИ95, доля
кадров в допуске, квантили 90/95 %.
"""
import os
import csv
import math

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import settings as S

REQ_DEG = 5.0            # требование по ошибке курса
P = 180.0


def _cstd(a, period=P):
    z = np.exp(1j * np.deg2rad(np.asarray(a) * 360.0 / period))
    R = abs(z.mean())
    return math.degrees(math.sqrt(max(-2 * math.log(max(R, 1e-12)),
                                      0))) * period / 360


def _cmean(a, period=P):
    z = np.exp(1j * np.deg2rad(np.asarray(a) * 360.0 / period))
    return (np.degrees(np.angle(z.mean())) * period / 360) % period


def _dev(a, c, period=P):
    return ((np.asarray(a) - c + period / 2) % period) - period / 2


def _circ_median(vals, period):
    v = np.asarray(vals) % period
    best, best_s = None, None
    for c in v:
        d = np.abs(((v - c + period / 2) % period) - period / 2)
        s_ = d.sum()
        if best_s is None or s_ < best_s:
            best, best_s = c, s_
    return float(best)


def tfilter(y, period, alt):
    out = np.empty_like(y)
    out[0] = y[0]
    flips = 0
    for i in range(1, len(y)):
        c = (y[i], (y[i] + alt) % period)
        d = [abs(((v - out[i - 1] + period / 2) % period) - period / 2)
             for v in c]
        j = int(np.argmin(d))
        out[i] = c[j]
        flips += j
    return out, flips


def anchors_from_ckpt():
    """{session_norm: A0_deg} из чекпойнта дообучения, иначе {}."""
    try:
        import torch
        ck = torch.load(S.NAV_CKPT, map_location='cpu')
        anc = ck.get('finetune', {}).get('anchors_deg') or {}
        return {k.replace('/', '_').replace(os.sep, '_'): float(v)
                for k, v in anc.items()}
    except Exception:
        return {}


def _boot_ci(x, fn, n=1000, seed=0):
    rng = np.random.default_rng(seed)
    x = np.asarray(x)
    vals = [fn(x[rng.integers(0, len(x), len(x))]) for _ in range(n)]
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def build(csv_path=None, out_jpg=None):
    global P
    csv_path = csv_path or S.NAV_CSV
    out_jpg = out_jpg or os.path.join(S.OUT_DIR, 'nav_yaw_vs_bearing.jpg')
    rows_all = [r for r in csv.DictReader(open(csv_path, encoding='utf-8'))
                if r.get('bearing_deg') and r.get('yaw_axis')]
    n_edge = sum(1 for r in rows_all if r.get('bearing_edge') == '1')
    n_unc = sum(1 for r in rows_all if r.get('bearing_confirmed') == '0')
    rows = [r for r in rows_all
            if not (S.BEARING_EXCLUDE_EDGE
                    and r.get('bearing_edge', '') == '1')
            and not (S.BEARING_REQUIRE_DETECT
                     and r.get('bearing_confirmed', '') == '0')]
    print(f'[report] кадров с опорой {len(rows_all)} (рамка у края: '
          f'{n_edge}, без подтверждения детектором: {n_unc}); '
          f'исключено {len(rows_all) - len(rows)}; в оценке {len(rows)}')
    if len(rows) < 10:
        print('[report] мало кадров с целью — фигура не строится')
        return None
    absolute = all(r.get('yaw_mode', '') == 'absolute' for r in rows)
    P = 360.0 if absolute else 180.0
    ycol = 'yaw' if absolute else 'yaw_axis'
    print(f"[report] режим курса: "
          f"{'абсолютный (mod 360)' if absolute else 'ось (mod 180)'}; "
          f"столбец {ycol}")
    y = np.array([float(r[ycol]) for r in rows])
    b = np.array([float(r['bearing_deg']) for r in rows])
    sess = np.array([r['session'].replace('/', '_') for r in rows])

    # опора: якорь чекпойнта или медиана (yaw + beta) по сессиям
    anc = anchors_from_ckpt()
    a0 = np.empty(len(rows))
    src = []
    for s_ in np.unique(sess):
        m = sess == s_
        if s_ in anc:
            a0[m] = anc[s_]
            src.append(f'{s_}: {anc[s_]:.1f}° (чекпойнт)')
        else:
            v = _circ_median((y[m] + b[m]) % P, P)
            a0[m] = v
            src.append(f'{s_}: {v:.1f}° (медиана CSV)')
    print('[report] якорь опоры: ' + '; '.join(src))
    err = _dev(y - (a0 - b), 0.0, P)         # знаковая ошибка курса
    ae = np.abs(err)
    med = float(np.median(ae))
    lo, hi = _boot_ci(ae, np.median)
    share = float((ae <= REQ_DEG).mean())
    slo, shi = _boot_ci(ae, lambda x: (x <= REQ_DEG).mean())
    q90, q95 = np.percentile(ae, 90), np.percentile(ae, 95)
    sig = float(np.std(err))
    print(f'[report] ошибка курса vs опора: медиана {med:.2f}° '
          f'[ДИ95 {lo:.2f}; {hi:.2f}], σ {sig:.2f}°, доля |ошибка| ≤ '
          f'{REQ_DEG:.0f}°: {share:.1%} [ДИ95 {slo:.1%}; {shi:.1%}], '
          f'P90 {q90:.1f}°, P95 {q95:.1f}°')

    c0 = _cmean(y)
    yd = _dev(y, c0)
    slope, inter = np.polyfit(b, yd, 1)
    cent, sig_b, cnt = [], [], []
    for lo_ in np.arange(-22, 22.0, 2.0):
        m = (b >= lo_) & (b < lo_ + 2.0)
        if m.sum() >= 10:
            cent.append(lo_ + 1.0)
            sig_b.append(_cstd(y[m], P))
            cnt.append(int(m.sum()))

    fig, ax = plt.subplots(3, 2, figsize=(12.5, 12.6))
    a = ax[0, 0]
    a.scatter(b, y, s=8, alpha=0.5)
    bb = np.linspace(b.min(), b.max(), 50)
    a.plot(bb, (c0 + inter + slope * bb) % P, 'r-', lw=2,
           label=f'аппроксимация: наклон {slope:+.2f}')
    a.set_xlabel('угол на цель β, град (вправо +)')
    a.set_ylabel(f'курс, град (mod {int(P)})')
    a.grid(alpha=0.3)
    a.legend(fontsize=9)
    a.set_title('а) курс от угла на цель', fontsize=10)

    a = ax[0, 1]
    if cent:
        a.bar(cent, sig_b, width=1.7)
        for x_, s_, n_ in zip(cent, sig_b, cnt):
            a.text(x_, s_ + 0.2, str(n_), ha='center', fontsize=7)
        a.axhline(float(np.median(sig_b)), color='r', ls='--', lw=1,
                  label=f'медиана σ = {np.median(sig_b):.1f}°')
        a.legend(fontsize=9)
    a.set_xlabel('угол на цель β, град (бины 2°; числа — кадров)')
    a.set_ylabel('σ курса в бине, град')
    a.grid(alpha=0.3)
    a.set_title('б) устойчивость при одинаковом угле на цель', fontsize=10)

    a = ax[1, 0]
    a.hist(y, bins=36, color='#2ca02c', alpha=0.85)
    a.set_xlabel(f'курс, град (mod {int(P)})')
    a.set_ylabel('кадров')
    a.grid(alpha=0.3)
    half = P / 2
    a.set_title(f'в) распределение курса: circ-σ {_cstd(y, P):.1f}° '
                f'(mod {int(half)}: {_cstd(y, half):.1f}°)', fontsize=10)

    a = ax[1, 1]
    n = len(y)
    if absolute:
        yu = np.rad2deg(np.unwrap(np.deg2rad(y)))
        w = 9
        low = np.convolve(yu, np.ones(w) / w, mode='valid')
        xs = np.arange(w // 2, w // 2 + len(low))
        a.plot(np.arange(n), yu - yu[0], '.', ms=3, alpha=0.6,
               label='развёрнутый курс')
        a.plot(xs, low - yu[0], 'r-', lw=1.2, label='сглаживание (окно 3 с)')
        a.set_ylabel('накопленный курс, град')
        a.set_title('г) развёрнутая траектория', fontsize=10)
        wind = float(low.max() - low.min())
        loc = float(np.std(yu[xs] - low))
        print(f'[report] absolute: размах накрутки {wind:.0f}°; '
              f'локальная σ (окно 3 с) {loc:.1f}°'
              + (' — накрутка много больше 360°: выход модели дрейфует'
                 if wind > 400 else ''))
    else:
        yf, fl = tfilter(y, P, P / 2)
        a.plot(np.arange(n), yd, '.', ms=3, alpha=0.5,
               label='курс − circ-среднее (сырой)')
        a.plot(np.arange(n), _dev(yf, _cmean(yf)), '.', ms=3, alpha=0.6,
               color='#d62728', label='после фильтра перескоков')
        a.set_ylabel('отклонение курса, град')
        a.set_title(f'г) ход по времени: фильтр перескоков {int(P/2)}°',
                    fontsize=10)
        d1 = np.abs(((np.diff(yf) + P / 2) % P) - P / 2)
        print(f'[report] фильтр {int(P/2)}°: σ до {_cstd(y, P):.1f}° '
              f'-> после {_cstd(yf, P):.1f}°; перевёрнуто {fl}; '
              f'скачков >45° после: {int((d1 > 45).sum())}')
    a.set_xlabel('номер кадра с целью (по времени)')
    a.grid(alpha=0.3)
    a.legend(fontsize=9)

    # д) гистограмма ошибки курса с акцентом на ±5°
    a = ax[2, 0]
    lim = max(REQ_DEG * 3, float(np.percentile(ae, 99)) + 1)
    bins = np.linspace(-lim, lim, 41)
    a.hist(np.clip(err, -lim, lim), bins=bins, color='#1f77b4', alpha=0.85)
    a.axvspan(-REQ_DEG, REQ_DEG, color='green', alpha=0.10,
              label=f'допуск ±{REQ_DEG:.0f}°: {share:.0%} кадров '
                    f'[ДИ95 {slo:.0%}–{shi:.0%}]')
    a.axvline(-REQ_DEG, color='green', ls='--', lw=1.2)
    a.axvline(REQ_DEG, color='green', ls='--', lw=1.2)
    a.axvline(float(np.median(err)), color='r', lw=1.5,
              label=f'медиана |ошибки| {med:.2f}° [ДИ95 {lo:.2f}–{hi:.2f}]')
    a.set_xlabel('ошибка курса относительно опоры A0 − β, град')
    a.set_ylabel('кадров')
    a.grid(alpha=0.3)
    a.legend(fontsize=8, loc='upper right')
    a.set_title(f'д) ошибка курса: σ {sig:.1f}°, P90 {q90:.1f}°, '
                f'P95 {q95:.1f}° (n={n})', fontsize=10)

    # е) ошибка по кадрам с полосой ±5°
    a = ax[2, 1]
    idx = np.arange(n)
    inside = ae <= REQ_DEG
    a.axhspan(-REQ_DEG, REQ_DEG, color='green', alpha=0.10)
    a.axhline(REQ_DEG, color='green', ls='--', lw=1.2)
    a.axhline(-REQ_DEG, color='green', ls='--', lw=1.2,
              label=f'допуск ±{REQ_DEG:.0f}°')
    a.plot(idx[inside], err[inside], '.', ms=3.5, color='#1f77b4',
           label=f'в допуске: {inside.sum()} кадров')
    a.plot(idx[~inside], err[~inside], 'o', ms=3.5, color='#d62728',
           label=f'вне допуска: {(~inside).sum()} кадров')
    a.axhline(0, color='k', lw=0.6)
    a.set_ylim(-lim, lim)
    a.set_xlabel('номер кадра с целью (по времени)')
    a.set_ylabel('ошибка курса, град')
    a.grid(alpha=0.3)
    a.legend(fontsize=8, loc='upper right')
    a.set_title('е) ошибка курса по кадрам', fontsize=10)

    fig.suptitle(f'Кадров с целью: {n}; курс — nav_pol_resnet '
                 f'({"абсолютный" if absolute else "ось, mod 180"}); '
                 f'β — из ручной разметки; требование |ошибка| ≤ '
                 f'{REQ_DEG:.0f}°', fontsize=10)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out_jpg, dpi=140)
    plt.close(fig)
    print(f'[report] фигура: {out_jpg}')
    return out_jpg


if __name__ == '__main__':
    build()
