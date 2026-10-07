#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""cameras_utils_v3 — обработка кадров поляризационной (Sony IMX250MZR) и RGB камер.

Изменения относительно cameras_utils (v1):
  1. Все функции внутренне приводят вход к float32 — защита от переполнения
     uint8 при вычитании (в v1 латентный баг).
  2. S0 = (I0 + I45 + I90 + I135) / 2 — используются все 4 отсчёта
     (меньше шум), без смещения «+1».
  3. pol_intensity = sqrt(S1^2 + S2^2) — без «+1» под корнем.
  4. Нули в S1/S2 не подменяются; защищается только знаменатель DoLP
     через np.maximum(s0, eps).
  5. Исправлены докстринги (диапазон AoLP: (-pi/2, pi/2]).
  6. Визуализации: перцентильная нормализация вместо min/max,
     абсолютная шкала DoLP для сопоставимости кадров датасета.
  7. Аналитический Шехнер: Imin/Imax = (S0 -+ sqrt(S1^2+S2^2)) / 2
     вместо дискретного min/max по 4 отсчётам; возвращается карта
     пропускания t (физический прокси дальности).
  8. Дебайеризация: COLOR_BayerBG2BGR — подтверждено реальными цветами
     сцены (синие 3D-печатные макеты, телесный цвет кожи).

Раскладка мозаики IMX250MZR (подтверждена пользователем):
    (0,0)=90°  (0,1)=45°
    (1,0)=135° (1,1)=0°
"""
from dataclasses import dataclass
from typing import Tuple

import numpy as np
from numpy.typing import NDArray
import cv2

EPS = 1e-6


# ──────────────────────────────────────────────────────────────
# Базовая физика: Стокс, DoLP, AoLP
# ──────────────────────────────────────────────────────────────

def split_raw_pol(image: NDArray) -> NDArray:
    """Разбить сырой кадр (h, w) на 4 угловых канала (h/2, w/2, 4) float32.

    Порядок каналов в выходе: (0°, 45°, 90°, 135°).
    """
    image = np.asarray(image).squeeze()
    if image.ndim != 2:
        raise ValueError('Ожидается 2D сырой кадр поляризационной камеры.')
    img = image.astype(np.float32, copy=False)
    ch_90 = img[::2, ::2]
    ch_45 = img[::2, 1::2]
    ch_135 = img[1::2, ::2]
    ch_0 = img[1::2, 1::2]
    return np.stack((ch_0, ch_45, ch_90, ch_135), axis=2)


def calc_stokes(channels: NDArray) -> Tuple[NDArray, NDArray, NDArray]:
    """Параметры Стокса из 4 угловых каналов (h, w, 4) в порядке (0,45,90,135).

    S0 = (I0+I45+I90+I135)/2;  S1 = I0-I90;  S2 = I45-I135.
    Возвращает (S0, S1, S2), float32.
    """
    ch = np.asarray(channels, dtype=np.float32)
    c0, c45, c90, c135 = ch[..., 0], ch[..., 1], ch[..., 2], ch[..., 3]
    s0 = (c0 + c45 + c90 + c135) * 0.5
    s1 = c0 - c90
    s2 = c45 - c135
    return s0, s1, s2


def calc_aolp(s1: NDArray, s2: NDArray) -> NDArray:
    """Угол линейной поляризации, рад: AoLP = 0.5*atan2(S2, S1).

    Диапазон (-pi/2, pi/2], период pi.
    """
    return 0.5 * np.arctan2(s2, s1)


def calc_pol_intensity(s1: NDArray, s2: NDArray) -> NDArray:
    """Поляризованная интенсивность sqrt(S1^2 + S2^2)."""
    return np.hypot(np.asarray(s1, np.float32), np.asarray(s2, np.float32))


def calc_dolp(s0: NDArray, pol_int: NDArray, eps: float = EPS) -> NDArray:
    """Степень линейной поляризации DoLP = pol_int / S0, обрезка в [0, 1]."""
    return np.clip(pol_int / np.maximum(s0, eps), 0.0, 1.0)


@dataclass
class PolFrame:
    """Все производные величины одного поляризационного кадра (float32)."""
    channels: NDArray  # (h, w, 4) в порядке (0,45,90,135)
    s0: NDArray
    s1: NDArray
    s2: NDArray
    pol_int: NDArray
    dolp: NDArray
    aolp: NDArray


def stokes_from_raw(raw: NDArray) -> PolFrame:
    """Полный расчёт из сырого кадра: split -> Стокс -> DoLP/AoLP."""
    ch = split_raw_pol(raw)
    s0, s1, s2 = calc_stokes(ch)
    pint = calc_pol_intensity(s1, s2)
    return PolFrame(channels=ch, s0=s0, s1=s1, s2=s2, pol_int=pint,
                    dolp=calc_dolp(s0, pint), aolp=calc_aolp(s1, s2))


# ──────────────────────────────────────────────────────────────
# Аналитический Шехнер: descattering + карта пропускания
# ──────────────────────────────────────────────────────────────

def analytic_min_max(s0: NDArray, pol_int: NDArray) -> Tuple[NDArray, NDArray]:
    """Аналитические экстремумы синусоиды Малюса из Стокса.

    I(theta) = (S0 + S1*cos2theta + S2*sin2theta)/2  =>
    Imin = (S0 - |P|)/2,  Imax = (S0 + |P|)/2,  |P| = sqrt(S1^2+S2^2).
    Точнее дискретного min/max по 4 отсчётам (те смещены вверх/вниз).
    """
    i_min = np.clip((s0 - pol_int) * 0.5, 0.0, None)
    i_max = (s0 + pol_int) * 0.5
    return i_min, i_max


def schechner_descatter(
    i_min: NDArray,
    i_max: NDArray,
    t_min: float = 0.2,
    bg_percentile: float = 95.0,
    robust_norm: bool = True,
) -> Tuple[NDArray, NDArray]:
    """Удаление обратного рассеяния по Шехнеру.

    Imin ~ фоновая засветка B (рассеяние поляризовано), Imax - Imin ~ сигнал.
    t = 1 - Imin/B_global — карта пропускания (прокси дальности:
    меньше t -> больше воды на луче).

    Returns
    -------
    (J, t): J — восстановленный сигнал float32 в [0, 1];
            t — карта пропускания float32 в [t_min, 1].
    """
    b_global = max(float(np.percentile(i_min, bg_percentile)), EPS)
    t = 1.0 - np.clip(i_min / b_global, 0.0, 1.0)
    t = np.clip(t, t_min, 1.0).astype(np.float32)
    j = (i_max - i_min) / t
    top = np.percentile(j, 99.9) if robust_norm else j.max()
    j = np.clip(j / max(float(top), EPS), 0.0, 1.0)
    return j.astype(np.float32), t


def apply_clahe(gray_uint8: NDArray, clip_limit: float = 3.0, tile: int = 8) -> NDArray:
    """CLAHE для grayscale uint8."""
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile, tile))
    return clahe.apply(gray_uint8)


def enhance_from_stokes(
    pf: PolFrame,
    t_min: float = 0.2,
    bg_percentile: float = 95.0,
    clahe_clip: float = 3.0,
) -> Tuple[NDArray, NDArray]:
    """Полное подводное улучшение: аналитический Шехнер + CLAHE.

    Returns
    -------
    (enh_uint8, t_map): улучшенное grayscale uint8 и карта пропускания float32.
    """
    i_min, i_max = analytic_min_max(pf.s0, pf.pol_int)
    j, t_map = schechner_descatter(i_min, i_max, t_min=t_min,
                                   bg_percentile=bg_percentile)
    enh = apply_clahe((j * 255.0).astype(np.uint8), clip_limit=clahe_clip)
    return enh, t_map


# ──────────────────────────────────────────────────────────────
# RGB
# ──────────────────────────────────────────────────────────────

def debayer_rgb(raw_bayer: NDArray) -> NDArray:
    """Дебайеризация сырого RGB-кадра -> BGR uint8 (для cv2.imwrite).

    Паттерн COLOR_BayerBG2BGR подтверждён по реальным цветам сцены
    (синие 3D-печатные аквалангист/АНПА, телесный цвет кожи рук).
    """
    return cv2.cvtColor(np.asarray(raw_bayer), cv2.COLOR_BayerBG2BGR)


def pol_pseudo_bgr(channels: NDArray) -> NDArray:
    """Псевдоцвет из трёх угловых каналов поляризации (ревизия 2).

    Отображение: R = I(0°), G = I(90°), B = I(135°); канал 45° не используется.
    Абсолютная шкала 0..255 без нормализации: неполяризованные участки
    серые (R≈G≈B), поляризация даёт цветовой сдвиг (оттенок ~ AoLP,
    насыщенность ~ DoLP), яркость сохраняет сцену.

    Parameters
    ----------
    channels : NDArray
        (h, w, 4) в порядке (0, 45, 90, 135), как из split_raw_pol.

    Returns
    -------
    NDArray
        BGR uint8 (h, w, 3) для cv2.imwrite.
    """
    ch = np.asarray(channels, dtype=np.float32)
    r, g, b = ch[..., 0], ch[..., 2], ch[..., 3]
    return np.clip(np.stack((b, g, r), axis=2), 0.0, 255.0).astype(np.uint8)


# ──────────────────────────────────────────────────────────────
# Визуализации (сопоставимые по датасету)
# ──────────────────────────────────────────────────────────────

def normalize_percentile(a: NDArray, lo: float = 1.0, hi: float = 99.0) -> NDArray:
    """Нормализация в uint8 по перцентилям кадра (устойчива к бликам)."""
    p_lo, p_hi = np.percentile(a, [lo, hi])
    span = max(float(p_hi - p_lo), EPS)
    x = np.clip((a - p_lo) / span, 0.0, 1.0)
    return (x * 255.0).astype(np.uint8)


def dolp_to_uint8(dolp: NDArray, dolp_max: float = 0.35) -> NDArray:
    """DoLP в абсолютной шкале [0, dolp_max] -> uint8 (одинакова для всех кадров)."""
    return (np.clip(dolp / dolp_max, 0.0, 1.0) * 255.0).astype(np.uint8)


def aolp_to_bgr(aolp: NDArray, dolp: NDArray = None,
                dolp_max: float = 0.35) -> NDArray:
    """AoLP -> цвет: H = угол (pi-периодичность = wrap hue), S = 255,
    V = DoLP в абсолютной шкале (или 255, если dolp не задан)."""
    h = np.rint((aolp + np.pi / 2) / np.pi * 179.0).astype(np.uint8)
    s = np.full_like(h, 255)
    v = dolp_to_uint8(dolp, dolp_max) if dolp is not None else np.full_like(h, 255)
    return cv2.cvtColor(cv2.merge((h, s, v)), cv2.COLOR_HSV2BGR)


def hsv_pol(aolp: NDArray, dolp: NDArray, s0: NDArray,
            dolp_max: float = 0.35) -> NDArray:
    """Композит: H = AoLP, S = DoLP (абс. шкала), V = S0 (перцентильная норм.)."""
    h = np.rint((aolp + np.pi / 2) / np.pi * 179.0).astype(np.uint8)
    s = dolp_to_uint8(dolp, dolp_max)
    v = normalize_percentile(s0, 1.0, 99.5)
    return cv2.cvtColor(cv2.merge((h, s, v)), cv2.COLOR_HSV2BGR)


# ──────────────────────────────────────────────────────────────
# Вторая волна представлений поляризации (v3; rev2: + i_dolp)
# ──────────────────────────────────────────────────────────────

def pol_stokes_bgr(channels: NDArray) -> NDArray:
    """Абсолютное представление Стокса: R = S0, G = S1, B = S2 -> BGR.

    S1 и S2 растягиваются ОБЩИМ симметричным масштабом по перцентилю |S|,
    поэтому сохраняется абсолютная величина поляризации: слабо
    поляризованный фон остаётся серым (128), сильно поляризованные
    поверхности уходят в цвет. Разрыва угла нет — направление закодировано
    парой (S1, S2) непрерывно (сырой AoLP рвётся при переходе ±90°).

    Parameters
    ----------
    channels : NDArray
        (h, w, 4) в порядке (0, 45, 90, 135), как из split_raw_pol.

    Returns
    -------
    NDArray
        BGR uint8 (h, w, 3).
    """
    ch = np.asarray(channels, dtype=np.float32)
    s0, s1, s2 = calc_stokes(ch)
    # общий масштаб: 99-й перцентиль |S1|,|S2|, но не меньше 2% от медианы
    # S0 — иначе в слабо поляризованной сцене шум растягивается на всю шкалу
    scale = max(float(np.percentile(np.abs(np.stack((s1, s2))), 99.0)),
                0.02 * float(np.median(s0)), EPS)
    g = np.clip(s1 / scale, -1.0, 1.0) * 127.0 + 128.0
    b = np.clip(s2 / scale, -1.0, 1.0) * 127.0 + 128.0
    r = normalize_percentile(s0).astype(np.float32)
    return np.clip(np.stack((b, g, r), axis=2), 0, 255).astype(np.uint8)


def pol_idolp_bgr(channels: NDArray) -> NDArray:
    """Интенсивность + самонормированное кодирование угла -> BGR.

    R = I (растянутая интенсивность), G = DoLP*cos(2*AoLP) = S1/S0,
    B = DoLP*sin(2*AoLP) = S2/S0, оба с центром 128. В отличие от
    pol_stokes_bgr нормировка ПОКАДРОВО-относительная (деление на S0):
    контраст поляризации не зависит от освещённости пикселя, а взвешивание
    на DoLP подавляет шум угла там, где поляризация слаба.

    Parameters
    ----------
    channels : NDArray
        (h, w, 4) в порядке (0, 45, 90, 135).

    Returns
    -------
    NDArray
        BGR uint8 (h, w, 3).
    """
    ch = np.asarray(channels, dtype=np.float32)
    s0, s1, s2 = calc_stokes(ch)
    den = np.maximum(s0, EPS)
    g = np.clip(s1 / den, -1.0, 1.0) * 127.0 + 128.0
    b = np.clip(s2 / den, -1.0, 1.0) * 127.0 + 128.0
    r = normalize_percentile(s0).astype(np.float32)
    return np.clip(np.stack((b, g, r), axis=2), 0, 255).astype(np.uint8)


def pol_i_dolp_bgr(channels: NDArray) -> NDArray:
    """Ранняя фьюжн I+DoLP БЕЗ угла: R = I, G = DoLP, B = I*(1-DoLP).

    Абляционный канал для декомпозиции вклада поляризационных величин:
    s0 (только I) -> i_dolp (+степень поляризации) -> idolp (+угол).
    Вся информация входа — {I, DoLP}; третий канал — неполяризованная
    составляющая интенсивности, производная от тех же величин (новой
    информации не добавляет, но делает все три канала содержательными).
    DoLP — в абсолютной шкале [0, 0.35], как в канале dolp.

    Parameters
    ----------
    channels : NDArray
        (h, w, 4) в порядке (0, 45, 90, 135), как из split_raw_pol.

    Returns
    -------
    NDArray
        BGR uint8 (h, w, 3).
    """
    ch = np.asarray(channels, dtype=np.float32)
    s0, s1, s2 = calc_stokes(ch)
    dolp = np.sqrt(s1 * s1 + s2 * s2) / np.maximum(s0, EPS)
    r = normalize_percentile(s0).astype(np.float32)
    g = np.clip(dolp / 0.35, 0.0, 1.0) * 255.0
    b = r * np.clip(1.0 - dolp, 0.0, 1.0)
    return np.clip(np.stack((b, g, r), axis=2), 0, 255).astype(np.uint8)
