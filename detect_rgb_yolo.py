#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""detect_rgb_yolo — опорный канал обнаружения по цветной камере
(модуль архитектуры)."""
import settings as S
from detect_common import UltralyticsDetector


def make():
    return UltralyticsDetector(S.YOLO_WEIGHTS['rgb'])
