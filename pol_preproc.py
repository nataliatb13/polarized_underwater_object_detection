#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""pol_preproc — предобработка поляризационных данных (модуль архитектуры).

Из сырого .npy поляризационной камеры по четырём интенсивностям
суперпикселя вычисляются параметры Стокса и формируется представление
idolp (функции — из проверенного cameras_utils_v3 корня проекта).
Кадры цветной камеры дебайеризуются тем же модулем.

Псевдо-HSV для навигации здесь НЕ строится: модуль nav_pol_resnet
загружает .npy собственным конвейером — гарантия идентичности входа
обученной модели.
"""
import os
import sys

import numpy as np
import cv2

import settings as S

sys.path.insert(0, S.ROOT)
import cameras_utils_v3 as cu                     # noqa: E402


def pol_idolp(npy_path):
    raw = np.load(npy_path)
    ch4 = cu.split_raw_pol(raw)
    return cu.pol_idolp_bgr(ch4)


def rgb_bgr(npy_path, out_w=None):
    raw = np.load(npy_path)
    bgr = cu.debayer_rgb(raw)
    if out_w and bgr.shape[1] != out_w:
        h = int(bgr.shape[0] * out_w / bgr.shape[1])
        bgr = cv2.resize(bgr, (out_w, h))
    return bgr
