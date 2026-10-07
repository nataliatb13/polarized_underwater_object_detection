#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""detect_pol_yolo — обнаружение объектов интереса по каналу idolp
(модуль архитектуры; обёртка над detect_common с весами
перекрёстного режима turbid2sunny)."""
import settings as S
from detect_common import UltralyticsDetector


def make():
    return UltralyticsDetector(S.YOLO_WEIGHTS['idolp'])
