# -*- coding: utf-8 -*-
"""Evaluation COCO unifiee : les 7 modeles sont notes par le MEME code.

Chaque runner produit un fichier de detections au format COCO
(`[{image_id, category_id, bbox: [x, y, w, h], score}, ...]`) sur le fold de
validation, evalue ensuite contre `cocoify.val_gt_json(fold)`.

Deux familles de metriques :
  - mAP@0.5 et mAP@0.5:0.95 par classe (pycocotools, maxDets = cfg.MAX_DET) ;
  - TP / FP / FN puis P / R / F1 par classe, a conf >= cfg.CONF_PR et
    IoU >= cfg.IOU_PR, apparies gloutonnement par score decroissant
    (meme convention que la matrice de confusion d'Ultralytics).
"""

import contextlib
import io
import json
from pathlib import Path

import numpy as np

from . import config as cfg


def _silent(fn, *args, **kwargs):
    """Execute fn en avalant les impressions de pycocotools."""
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        return fn(*args, **kwargs)


def _load(path_or_obj):
    if isinstance(path_or_obj, (str, Path)):
        return json.loads(Path(path_or_obj).read_text())
    return path_or_obj


# ------------------------------------------------------------------- mAP COCO
def _ap(cocoeval, iou_thr=None):
    """AP moyenne lue directement dans la matrice de precision accumulee.

    `COCOeval.summarize()` n'est pas utilisable ici : son `stats[0]` est calcule
    avec `maxDets=100` code en dur et renvoie -1 des que `params.maxDets` ne
    contient pas 100. On refait donc le meme calcul que `_summarize`, mais au
    dernier `maxDets` configure (cfg.MAX_DET).

    Renvoie None si aucune valeur valide (classe absente du ground-truth).
    """
    prec = cocoeval.eval["precision"]           # [IoU, rappel, classe, aire, maxDets]
    p = cocoeval.params
    aind = [i for i, a in enumerate(p.areaRngLbl) if a == "all"][0]
    mind = len(p.maxDets) - 1                   # dernier maxDets = cfg.MAX_DET
    if iou_thr is None:
        s = prec[:, :, :, aind, mind]
    else:
        t = int(np.argmin(np.abs(np.array(p.iouThrs) - iou_thr)))
        s = prec[t:t + 1, :, :, aind, mind]
    valid = s[s > -1]
    return float(np.mean(valid)) if valid.size else None


def coco_eval_per_class(gt_json, dt_json, ncl=None, max_dets=None):
    """{classe -> {"map50", "map5095"}} par classe, via pycocotools.

    Une classe absente du ground-truth du fold renvoie NaN (et non 0), pour ne
    pas tirer la moyenne macro vers le bas.
    """
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval

    ncl = ncl or len(cfg.CLASS_NAMES)
    max_dets = max_dets or cfg.MAX_DET

    coco_gt = _silent(COCO, str(gt_json))
    n_gt = {ci: len(coco_gt.getAnnIds(catIds=[ci + 1])) for ci in range(ncl)}
    dets = _load(dt_json)
    if not dets:
        # aucune detection : AP nulle la ou il y a des objets a trouver
        return {ci: {"map50": 0.0 if n_gt[ci] else float("nan"),
                     "map5095": 0.0 if n_gt[ci] else float("nan")}
                for ci in range(ncl)}

    coco_dt = _silent(coco_gt.loadRes,
                      str(dt_json) if isinstance(dt_json, (str, Path)) else dets)

    out = {}
    for ci in range(ncl):
        e = COCOeval(coco_gt, coco_dt, "bbox")
        e.params.catIds = [ci + 1]                 # categories COCO 1-based
        e.params.maxDets = [1, 10, max_dets]
        _silent(e.evaluate)
        _silent(e.accumulate)
        fallback = 0.0 if n_gt[ci] else float("nan")
        ap = _ap(e)
        ap50 = _ap(e, iou_thr=0.5)
        out[ci] = {"map5095": fallback if ap is None else ap,
                   "map50": fallback if ap50 is None else ap50}
    return out


# --------------------------------------------------------------- TP / FP / FN
def _iou_xywh(a, b):
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    x1, y1 = max(ax, bx), max(ay, by)
    x2, y2 = min(ax + aw, bx + bw), min(ay + ah, by + bh)
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    union = aw * ah + bw * bh - inter
    return inter / union if union > 0 else 0.0


def pr_counts(gt_json, dt_json, ncl=None, conf_thr=None, iou_thr=None):
    """TP / FP / FN par classe, a conf et IoU fixes (appariement glouton)."""
    ncl = ncl or len(cfg.CLASS_NAMES)
    conf_thr = cfg.CONF_PR if conf_thr is None else conf_thr
    iou_thr = cfg.IOU_PR if iou_thr is None else iou_thr

    gt = _load(gt_json)
    dt = _load(dt_json)

    gt_by, dt_by = {}, {}
    for a in gt["annotations"]:
        gt_by.setdefault(a["image_id"], []).append((a["category_id"], a["bbox"]))
    for d in dt:
        dt_by.setdefault(d["image_id"], []).append((d["category_id"], d["bbox"], d["score"]))

    tp = [0] * ncl
    fp = [0] * ncl
    fn = [0] * ncl
    for img in set(gt_by) | set(dt_by):
        for c in range(ncl):
            cat = c + 1
            gts = [b for k, b in gt_by.get(img, []) if k == cat]
            dts = sorted([(b, s) for k, b, s in dt_by.get(img, [])
                          if k == cat and s >= conf_thr],
                         key=lambda t: -t[1])
            matched = set()
            for box, _score in dts:
                best_iou, best_j = 0.0, -1
                for j, g in enumerate(gts):
                    if j in matched:
                        continue
                    v = _iou_xywh(box, g)
                    if v > best_iou:
                        best_iou, best_j = v, j
                if best_iou >= iou_thr and best_j >= 0:
                    tp[c] += 1
                    matched.add(best_j)
                else:
                    fp[c] += 1
            fn[c] += len(gts) - len(matched)
    return tp, fp, fn


# ------------------------------------------------------------------ synthese
def evaluate_predictions(gt_json, dt_json, class_names=None):
    """Toutes les metriques d'un (modele, fold) sous forme de dict plat.

    Renvoie map50_macro, map5095_macro et, par classe :
    <classe>_map50, _map5095, _P, _R, _F1, _TP, _FP, _FN.
    """
    class_names = class_names or cfg.CLASS_NAMES
    ncl = len(class_names)
    names = [class_names[i] for i in range(ncl)]

    per = coco_eval_per_class(gt_json, dt_json, ncl)
    tp, fp, fn = pr_counts(gt_json, dt_json, ncl)

    row = {
        "map50_macro": round(float(np.nanmean([per[c]["map50"] for c in range(ncl)])), 4),
        "map5095_macro": round(float(np.nanmean([per[c]["map5095"] for c in range(ncl)])), 4),
    }
    for c, nm in enumerate(names):
        p = tp[c] / (tp[c] + fp[c]) if (tp[c] + fp[c]) else 0.0
        r = tp[c] / (tp[c] + fn[c]) if (tp[c] + fn[c]) else 0.0
        f1 = 2 * p * r / (p + r) if (p + r) else 0.0
        row[f"{nm}_map50"] = round(per[c]["map50"], 4)
        row[f"{nm}_map5095"] = round(per[c]["map5095"], 4)
        row[f"{nm}_P"] = round(p, 4)
        row[f"{nm}_R"] = round(r, 4)
        row[f"{nm}_F1"] = round(f1, 4)
        row[f"{nm}_TP"] = int(tp[c])
        row[f"{nm}_FP"] = int(fp[c])
        row[f"{nm}_FN"] = int(fn[c])
    return row


def write_detections(dets, path):
    """Ecrit les detections COCO (et renvoie le chemin)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dets))
    return path
