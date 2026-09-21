# -*- coding: utf-8 -*-
"""Conversion des folds YOLO en datasets COCO (deux arborescences).

Deux layouts sont produits a partir des MEMES listes de tuiles :

  layout="flat"  (RF-DETR / Roboflow)
      ds/train/_annotations.coco.json  + images
      ds/valid/_annotations.coco.json  + images
      ds/test/_annotations.coco.json   + images   (= valid, exige par rfdetr)

  layout="coco"  (RT-DETR, D-FINE, YOLOX)
      ds/train2017/ + ds/val2017/
      ds/annotations/instances_train2017.json
      ds/annotations/instances_val2017.json

Les images sont des liens symboliques : aucune copie, aucun octet duplique.

Convention de categories : COCO 1-based (`Apterous_aphid` = 1, `Alate_aphid` = 2),
soit `category_id = classe_yolo + 1`. Les identifiants d'images sont derives de
la liste triee des chemins, donc IDENTIQUES d'un framework a l'autre : le
ground-truth de validation d'un fold est le meme fichier pour tout le monde.
"""

import json
import shutil
from pathlib import Path

from . import config as cfg
from .folds import fold_train_paths, fold_val_paths, has_annotations, label_of

LAYOUTS = {
    "flat": {"train": "train", "val": "valid", "extra_val_copy": "test"},
    "coco": {"train": "train2017", "val": "val2017", "extra_val_copy": None},
}


def categories():
    """Categories COCO 1-based, dans l'ordre des classes YOLO."""
    return [{"id": i + 1, "name": n, "supercategory": "aphid"}
            for i, n in sorted(cfg.CLASS_NAMES.items())]


def _img_size(_path):
    """Toutes les tuiles font TILE_SIZE x TILE_SIZE (evite d'ouvrir les images)."""
    return cfg.TILE_SIZE, cfg.TILE_SIZE


def _yolo_boxes_px(lbl, W, H):
    """Lit un label YOLO normalise -> [(category_id, x, y, w, h)] en pixels."""
    out = []
    lbl = Path(lbl)
    if not lbl.exists():
        return out
    for line in lbl.read_text().splitlines():
        s = line.split()
        if len(s) < 5:
            continue
        c = int(float(s[0]))
        xc, yc, bw, bh = (float(v) for v in s[1:5])
        out.append((c + 1, (xc - bw / 2) * W, (yc - bh / 2) * H, bw * W, bh * H))
    return out


def coco_from_paths(paths):
    """Dict COCO a partir d'une liste de tuiles (ids derives du tri des chemins)."""
    images, anns, aid = [], [], 1
    for iid, ip in enumerate(sorted(set(str(p) for p in paths)), start=1):
        ip = Path(ip)
        W, H = _img_size(ip)
        images.append({"id": iid, "file_name": ip.name, "width": W, "height": H})
        for cat, x, y, w, h in _yolo_boxes_px(label_of(ip), W, H):
            anns.append({"id": aid, "image_id": iid, "category_id": cat,
                         "bbox": [round(x, 2), round(y, 2), round(w, 2), round(h, 2)],
                         "area": round(w * h, 2), "iscrowd": 0, "segmentation": []})
            aid += 1
    return {"images": images, "annotations": anns, "categories": categories()}


def val_gt_json(fold, force=False):
    """Ecrit (une fois) le ground-truth COCO du fold de validation et renvoie son chemin.

    C'est LE fichier de reference de l'evaluation unifiee : tous les modeles sont
    notes contre celui-ci, quel que soit leur framework.
    """
    out = Path(cfg.COCO_ROOT) / "gt" / f"val_fold{fold}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    if force or not out.exists():
        out.write_text(json.dumps(coco_from_paths(fold_val_paths(fold))))
    return out


def _link_all(paths, dst_dir):
    dst_dir = Path(dst_dir)
    dst_dir.mkdir(parents=True, exist_ok=True)
    for p in paths:
        p = Path(p)
        dst = dst_dir / p.name
        if not dst.exists():
            try:
                dst.symlink_to(p)
            except (FileExistsError, OSError):
                pass


def build_coco_fold(fold, layout="flat", neg_ratio=None, force=False, verbose=True):
    """Construit le dataset COCO d'un fold. Renvoie (dossier, n_pucerons, n_fonds)."""
    if layout not in LAYOUTS:
        raise ValueError(f"layout inconnu : {layout} (attendu {list(LAYOUTS)})")
    neg_ratio = cfg.NEG_RATIO if neg_ratio is None else neg_ratio

    train_paths = fold_train_paths(fold, neg_ratio)
    val_paths = fold_val_paths(fold)
    npos = sum(1 for p in train_paths if has_annotations(label_of(p)))
    nneg = len(train_paths) - npos

    names = LAYOUTS[layout]
    ds = Path(cfg.COCO_ROOT) / f"{layout}_fold{fold}_neg{neg_ratio}"
    marker = ds / ".done"
    if marker.exists() and not force:
        if verbose:
            print(f"    dataset COCO ({layout}) fold{fold} deja construit -> reutilise")
        return ds, npos, nneg
    if ds.exists():
        shutil.rmtree(ds)

    splits = [(names["train"], train_paths), (names["val"], val_paths)]
    if names["extra_val_copy"]:
        splits.append((names["extra_val_copy"], val_paths))

    for split_name, paths in splits:
        _link_all(paths, ds / split_name)
        doc = json.dumps(coco_from_paths(paths))
        if layout == "flat":
            (ds / split_name / "_annotations.coco.json").write_text(doc)
        else:
            ann_dir = ds / "annotations"
            ann_dir.mkdir(parents=True, exist_ok=True)
            (ann_dir / f"instances_{split_name}.json").write_text(doc)
        if verbose:
            print(f"    {split_name}: {len(paths)} images liees + COCO ecrit")

    marker.write_text("ok")
    return ds, npos, nneg


def coco_paths(ds, layout):
    """(dossier images train, json train, dossier images val, json val)."""
    names = LAYOUTS[layout]
    ds = Path(ds)
    if layout == "flat":
        return (ds / names["train"], ds / names["train"] / "_annotations.coco.json",
                ds / names["val"], ds / names["val"] / "_annotations.coco.json")
    return (ds / names["train"], ds / "annotations" / f"instances_{names['train']}.json",
            ds / names["val"], ds / "annotations" / f"instances_{names['val']}.json")
