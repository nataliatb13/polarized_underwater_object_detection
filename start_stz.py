#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""start_stz — оффлайн-оркестратор прогона серии 2 (чистая вода).

В боевой архитектуре модули выполняются параллельно в tmux-сессиях;
здесь тот же конвейер выполняется последовательно по кадрам с диска:

  file_camera_server -> pol_preproc -> detect_{pol,rgb}_yolo
                     -> nav_pol_resnet -> result_aggregator -> ИКИВ

Выход (Rodnik_17062026/detect_and_nav/):
  idolp/<stem>.jpg — idolp с рамками и углами ориентации (пр. верхний угол)
  rgb/<stem>.jpg   — rgb с рамками
  nav_session2.csv — углы по каждому кадру серии 2 + базовый угол на цель
  aggregated.csv   — решения «k из n» по каналам и классам
  ikiv_packets.jsonl — журнал пакетов ИКИВ

Требования боевого режима: torch, torchvision, pysolar, ultralytics;
рядом — nav_pol_resnet.py и чекпоинт nav_pol_resnet.pth.
Самотест без зависимостей: UDEPTH_MOCK=1.
"""
import os
import sys
import csv

import cv2

import settings as S
import file_camera_server as cam
import pol_preproc as pp
import nav_utils as nu
from detect_common import draw_dets, FakeDetector
from result_aggregator import KofN
from API_on_TCP import IKIV

MOCK = os.environ.get('UDEPTH_MOCK') == '1'


class FakeNav:
    def predict(self, npy_path, when=None):
        return {'roll': 1.0, 'pitch': -2.0, 'yaw': 123.4,
                'yaw_axis': 123.4 % 180, 'yaw_mode': 'mock',
                'sun_az': 180.0, 'sun_el': 45.0,
                'time_utc': '', 'sun_below_horizon': False,
                'target': 'mock'}


def make_workers():
    if MOCK:
        return FakeDetector(), FakeDetector(cls=1, conf=0.7), FakeNav()
    import detect_pol_yolo
    import detect_rgb_yolo
    import nav_pol_resnet as navmod
    nav = navmod.NavPolResNet(S.NAV_CKPT)
    print(f"[nav] чекпойнт: {S.NAV_CKPT}")
    crop = float(nav.ck.get('finetune', {}).get('crop_frac', 1.0) or 1.0)
    if crop < 1.0:
        import nav_crop
        navmod.frame_to_tensor = nav_crop.make_cropped_frame_to_tensor(
            navmod, crop)
        print(f"[nav] центральное кадрирование псевдо-HSV: доля {crop:.2f} "
              f"(из чекпойнта дообучения)")
    print(f"[nav] цель обучения: {nav.ck.get('target')}; "
          f"base={nav.ck.get('base')}, k={nav.ck.get('k')} "
          f"(k=1 -> абсолютный курс mod 360)")
    return detect_pol_yolo.make(), detect_rgb_yolo.make(), nav


def main():
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass
    det_pol, det_rgb, nav = make_workers()
    lab_idx = nu.build_label_index()
    split_idx = nu.build_split_index()
    z_idx = nu.build_z_index()
    agg = KofN()
    os.makedirs(S.OUT_DIR, exist_ok=True)
    ikiv = IKIV()
    os.makedirs(os.path.join(S.OUT_DIR, 'idolp'), exist_ok=True)
    os.makedirs(os.path.join(S.OUT_DIR, 'rgb'), exist_ok=True)
    rows = []
    agg_rows = []
    n = 0
    for fp in cam.iter_pairs():
        sess, stem, lp = lab_idx.get(fp.stem_ts, (fp.session, None, None))
        stem = stem or fp.ts.strftime('%Y%m%d_%H%M%S_%f')
        idolp = pp.pol_idolp(fp.pol_npy)
        rgb = pp.rgb_bgr(fp.rgb_npy, out_w=idolp.shape[1])
        d_pol = det_pol.predict(idolp)
        d_rgb = det_rgb.predict(rgb)
        try:
            navr = nav.predict(fp.pol_npy)
        except Exception as e:
            print(f'[nav][WARN] {stem}: {e!r}')
            navr = None
        beta, bcls, bbox = nu.bearing_from_label(lp, idolp.shape[1],
                                                 with_box=True)
        b_edge = int(nu.box_touches_edge(bbox)) if bbox else ''
        b_conf = (int(nu.box_confirmed(bbox, bcls, d_pol, idolp.shape[1],
                                       idolp.shape[0])) if bbox else '')
        draw_dets(idolp, d_pol)
        nu.overlay_nav(idolp, navr, n_objects=len(d_pol))
        draw_dets(rgb, d_rgb)
        cv2.imwrite(os.path.join(S.OUT_DIR, 'idolp', stem + '.jpg'),
                    idolp, [cv2.IMWRITE_JPEG_QUALITY, 90])
        cv2.imwrite(os.path.join(S.OUT_DIR, 'rgb', stem + '.jpg'),
                    rgb, [cv2.IMWRITE_JPEG_QUALITY, 90])
        zrow = z_idx.get((sess, stem), {})
        rows.append({
            'session': sess, 'stem': stem,
            'time': fp.ts.isoformat(),
            'split': split_idx.get(stem, ''),
            'roll': f"{navr['roll']:.2f}" if navr else '',
            'pitch': f"{navr['pitch']:.2f}" if navr else '',
            'yaw': f"{navr['yaw']:.2f}" if navr else '',
            'yaw_axis': f"{navr['yaw_axis']:.2f}" if navr else '',
            'yaw_mode': navr['yaw_mode'] if navr else '',
            'sun_az': f"{navr['sun_az']:.2f}" if navr else '',
            'sun_el': f"{navr['sun_el']:.2f}" if navr else '',
            'bearing_deg': f'{beta:.3f}' if beta is not None else '',
            'bearing_cls': S.CLASSES[bcls] if bcls is not None else '',
            'bearing_edge': b_edge,
            'bearing_confirmed': b_conf,
            'z_sub': zrow.get('z_sub', ''),
        })
        for channel, dets in (('idolp', d_pol), ('rgb', d_rgb)):
            for c, hits, dec in agg.update(channel, dets):
                agg_rows.append({'stem': stem, 'channel': channel,
                                 'cls': S.CLASSES[c], 'hits': hits,
                                 'n': agg.n, 'k': agg.k,
                                 'decision': dec})
        ikiv.send({'stem': stem, 'idolp': d_pol, 'rgb': d_rgb,
                   'nav': navr,
                   'agg': agg_rows[-2 * len(S.CLASSES):]})
        n += 1
        if n % 50 == 0:
            print(f'  обработано {n} пар')
    for path, data, fields in (
            (S.NAV_CSV, rows, list(rows[0].keys()) if rows else []),
            (S.AGG_CSV, agg_rows,
             list(agg_rows[0].keys()) if agg_rows else [])):
        if data:
            with open(path, 'w', newline='', encoding='utf-8') as f:
                w = csv.DictWriter(f, fieldnames=fields)
                w.writeheader()
                w.writerows(data)
    ikiv.close()
    print(f'Готово: пар {n}; кадры и CSV в {S.OUT_DIR}')
    try:
        import nav_report
        nav_report.build()
    except Exception as e:
        print(f'[report][WARN] фигура не построена: {e!r}')
    try:
        import detect_stats
        detect_stats.build()
    except Exception as e:
        print(f'[stats][WARN] статистика обнаружения не построена: {e!r}')
    if rows:
        with_nav = sum(1 for r in rows if r['yaw'])
        with_b = sum(1 for r in rows if r['bearing_deg'])
        print(f'  строк nav_csv: {len(rows)} (углы: {with_nav}, '
              f'базовый угол на цель: {with_b})')


if __name__ == '__main__':
    main()
