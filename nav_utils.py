#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""nav_utils — привязка навигации к разметке и оформление кадра.

Базовый угол на цель: beta = arctan(dx * PIX / (N * F)) — горизонтальный
азимут центра рамки относительно оптической оси (вправо — плюс), с
эффективным фокусом N_REFR * FOCAL_MM (та же оптическая модель, что в
фотограмметрической дальнометрии). Приоритет цели: submarine, затем
diver, затем shark.
"""
import os
import csv
import math

import cv2

import settings as S

PRIORITY = (1, 0, 2)


def build_label_index(include_turbid=False):
    """ts_compact -> (session, stem, label_path) по labels_full_v1/pol."""
    idx = {}
    root = S.LABELS_POL
    if not os.path.isdir(root):
        return idx
    for sess in os.listdir(root):
        if not include_turbid and S.TURBID_MARK in sess:
            continue
        d = os.path.join(root, sess)
        if not os.path.isdir(d):
            continue
        for f in os.listdir(d):
            if f.endswith('.txt'):
                stem = f[:-4]
                ts = stem.split('_', 1)[-1]
                idx[ts] = (sess, stem, os.path.join(d, f))
    return idx


def build_split_index():
    """stem -> split. Приоритет — splits.csv в корне набора (публикуемая
    форма); иначе — симлинки datasets_v2/<канал>/{train,val,test}."""
    out = {}
    sp = os.path.join(S.DATA, 'splits.csv')
    if os.path.exists(sp):
        for r in csv.DictReader(open(sp, encoding='utf-8')):
            out[r['stem']] = r['split']
        return out
    for split in ('train', 'val', 'test'):
        d = os.path.join(S.SPLIT_IMAGES, split, 'images')
        if not os.path.isdir(d):
            continue
        for f in os.listdir(d):
            if f.endswith('.jpg') and '__' in f:
                out[f[:-4].split('__', 1)[1]] = split
    return out


def build_z_index():
    out = {}
    if not os.path.exists(S.COND_CSV):
        return out
    for r in csv.DictReader(open(S.COND_CSV, encoding='utf-8')):
        if r.get('stream') == 'pol' and int(r.get('n_ref') or 0) >= 2:
            out[(r['session'], r['stem'])] = r
    return out


def bearing_from_label(label_path, img_w, with_box=False):
    """(beta_deg, cls[, box]) по крупнейшей рамке приоритетного класса;
    box = (cx, cy, w, h) нормированные."""
    none = (None, None, None) if with_box else (None, None)
    if not label_path or not os.path.exists(label_path):
        return none
    best = {}
    for ln in open(label_path, encoding='utf-8'):
        p = ln.split()
        if len(p) != 5:
            continue
        c = int(p[0])
        cx, cy, bw, bh = (float(p[1]), float(p[2]), float(p[3]),
                          float(p[4]))
        if c not in best or bw * bh > best[c][2] * best[c][3]:
            best[c] = (cx, cy, bw, bh)
    for c in PRIORITY:
        if c in best:
            cx = best[c][0]
            dx_px = (cx - 0.5) * img_w
            beta = math.degrees(math.atan(
                dx_px * S.PIX_MM_POL / (S.N_REFR * S.FOCAL_MM)))
            return (beta, c, best[c]) if with_box else (beta, c)
    return none


def box_touches_edge(box, margin=None):
    """Рамка (cx, cy, w, h норм.) ближе margin к границе кадра."""
    margin = S.BEARING_EDGE_MARGIN if margin is None else margin
    cx, cy, w, h = box
    return (cx - w / 2 < margin or cx + w / 2 > 1 - margin
            or cy - h / 2 < margin or cy + h / 2 > 1 - margin)


def box_confirmed(box, cls, dets, img_w, img_h, iou_min=None):
    """Есть ли детекция класса cls с IoU >= iou_min к рамке разметки."""
    iou_min = S.BEARING_IOU if iou_min is None else iou_min
    cx, cy, w, h = box
    lx1, ly1 = (cx - w / 2) * img_w, (cy - h / 2) * img_h
    lx2, ly2 = (cx + w / 2) * img_w, (cy + h / 2) * img_h
    la = max(lx2 - lx1, 0) * max(ly2 - ly1, 0)
    for c, conf, x1, y1, x2, y2 in dets:
        if c != cls:
            continue
        ix = max(0, min(lx2, x2) - max(lx1, x1))
        iy = max(0, min(ly2, y2) - max(ly1, y1))
        inter = ix * iy
        union = la + (x2 - x1) * (y2 - y1) - inter
        if union > 0 and inter / union >= iou_min:
            return True
    return False


def _font(size):
    from PIL import ImageFont
    try:
        import matplotlib
        return ImageFont.truetype(os.path.join(
            matplotlib.get_data_path(), 'fonts', 'ttf',
            'DejaVuSans-Bold.ttf'), size)
    except Exception:
        return ImageFont.load_default()


def overlay_nav(bgr, nav, n_objects=0, size=26):
    """Служебная надпись в правом верхнем углу (кириллица через PIL):
       Курс по поляризации: <yaw>°
       Объектов обнаружено: <N>"""
    import numpy as np
    from PIL import Image, ImageDraw
    lines = [f"Курс по поляризации: {nav['yaw']:.1f}°" if nav else
             "Курс по поляризации: —",
             f"Объектов обнаружено: {n_objects}"]
    pil = Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    d = ImageDraw.Draw(pil)
    font = _font(size)
    w = bgr.shape[1]
    y = 10
    for t in lines:
        tw = d.textlength(t, font=font)
        d.text((w - tw - 14, y), t, font=font, fill=(255, 255, 255),
               stroke_width=3, stroke_fill=(0, 0, 0))
        y += int(size * 1.35)
    bgr[:] = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)
    return bgr
