#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""detect_common — общий каркас модулей обнаружения.

UltralyticsDetector — боевой (YOLO11s, веса перекрёстного режима);
FakeDetector — самотестовая заглушка без torch.
"""
import cv2

import settings as S


class UltralyticsDetector:
    def __init__(self, weights):
        from ultralytics import YOLO
        self.model = YOLO(weights)
        print(f'[detect] веса: {weights}')

    def predict(self, bgr):
        """-> [(cls, conf, x1, y1, x2, y2)] в пикселях кадра."""
        r = self.model.predict(bgr, conf=S.CONF, imgsz=S.IMGSZ,
                               verbose=False)[0]
        out = []
        for b in r.boxes:
            x1, y1, x2, y2 = map(float, b.xyxy[0].tolist())
            out.append((int(b.cls[0]), float(b.conf[0]), x1, y1, x2, y2))
        return out


class FakeDetector:
    """Для самотеста: submarine в фиксированном месте каждого кадра."""

    def __init__(self, box=(0.30, 0.40, 0.50, 0.55), cls=1, conf=0.80):
        self.box, self.cls, self.conf = box, cls, conf

    def predict(self, bgr):
        h, w = bgr.shape[:2]
        x1, y1, x2, y2 = self.box
        return [(self.cls, self.conf, x1 * w, y1 * h, x2 * w, y2 * h)]


def draw_dets(bgr, dets):
    for c, conf, x1, y1, x2, y2 in dets:
        col = S.CLS_BGR.get(c, (200, 200, 200))
        cv2.rectangle(bgr, (int(x1), int(y1)), (int(x2), int(y2)), col, 2)
        txt = f'{S.CLASSES[c]} {conf:.2f}'
        ty = max(int(y1) - 6, 14)
        cv2.putText(bgr, txt, (int(x1), ty), cv2.FONT_HERSHEY_SIMPLEX,
                    0.55, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(bgr, txt, (int(x1), ty), cv2.FONT_HERSHEY_SIMPLEX,
                    0.55, col, 1, cv2.LINE_AA)
    return bgr
