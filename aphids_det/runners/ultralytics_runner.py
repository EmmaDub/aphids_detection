# -*- coding: utf-8 -*-
"""YOLO26n / YOLO11n / YOLO12n via Ultralytics.

Les hyperparametres restent ceux par defaut de chaque modele ; seules
l'augmentation (`augment.AUG`), les epoques, le batch et la taille d'image sont
imposees. Les metriques ne viennent PAS de `model.val()` mais de l'evaluateur
COCO unifie, comme pour les autres frameworks.
"""

import shutil
import time
from pathlib import Path

import yaml

from .. import bench, cocoify, config as cfg, evaluate
from ..augment import AUG, etat_albumentations, patch_ultralytics_visibility
from ..folds import fold_counts, fold_train_paths, fold_val_paths

PROJECT = "yolo_comp"


def build_fold_yaml(fold, neg_ratio=None):
    """Ecrit le dataset YAML d'un fold (train = liste figee, val = fold complet)."""
    neg_ratio = cfg.NEG_RATIO if neg_ratio is None else neg_ratio
    lines = fold_train_paths(fold, neg_ratio)

    # Copie LOCALE de la liste : Ultralytics lit mal les chemins raccourcis Drive.
    txt_local = Path(cfg.YAML_DIR) / f"train_fold{fold}_neg{neg_ratio}.txt"
    txt_local.parent.mkdir(parents=True, exist_ok=True)
    txt_local.write_text("\n".join(lines))

    ypath = Path(cfg.YAML_DIR) / f"compare_fold{fold}_neg{neg_ratio}.yaml"
    yaml.dump({"path": str(cfg.FOLDS_ROOT), "train": str(txt_local),
               "val": f"fold_{fold}/images", "names": cfg.CLASS_NAMES},
              open(ypath, "w"), sort_keys=False, allow_unicode=True)
    return ypath


def _epochs_info(trainer):
    """(meilleure epoque, derniere epoque) d'un entrainement Ultralytics."""
    import pandas as pd

    last = int(getattr(trainer, "epoch", -1))
    if last >= 0:
        last += 1
    best = getattr(trainer, "best_epoch", None)
    if best is None or best < 0:
        stopper = getattr(trainer, "stopper", None)
        best = getattr(stopper, "best_epoch", None) if stopper is not None else None
    if best is not None and best >= 0:
        return int(best), last
    try:
        d = pd.read_csv(Path(trainer.save_dir) / "results.csv")
        d.columns = [c.strip() for c in d.columns]
        col = next((c for c in d.columns if "mAP50(B)" in c), None)
        if col:
            return int(d[col].idxmax()) + 1, last
    except Exception:
        pass
    return -1, last


def predict_coco(model, images, gt_json, out_json):
    """Exporte les detections du fold de validation au format COCO."""
    import json

    gt = json.loads(Path(gt_json).read_text())
    name2id = {im["file_name"]: im["id"] for im in gt["images"]}
    paths = [p for p in images if Path(p).name in name2id]

    dets = []
    results = model.predict(source=paths, imgsz=cfg.IMGSZ, conf=cfg.CONF_EVAL,
                            iou=cfg.NMS_IOU, max_det=cfg.MAX_DET,
                            verbose=False, stream=True)
    for r in results:
        iid = name2id.get(Path(r.path).name)
        if iid is None or r.boxes is None:
            continue
        for box, cls_id, score in zip(r.boxes.xyxy.tolist(),
                                      r.boxes.cls.tolist(),
                                      r.boxes.conf.tolist()):
            x1, y1, x2, y2 = box
            dets.append({"image_id": iid,
                         "category_id": int(cls_id) + 1,   # COCO 1-based
                         "bbox": [round(x1, 2), round(y1, 2),
                                  round(x2 - x1, 2), round(y2 - y1, 2)],
                         "score": round(float(score), 5)})
    return evaluate.write_detections(dets, out_json)


def run_fold(modele, weights, fold):
    """Entraine et evalue un modele Ultralytics sur un fold."""
    from ultralytics import YOLO

    patch_ultralytics_visibility()      # seuil de visibilite des boites tronquees
    # Ultralytics ajoute Blur/MedianBlur/ToGray/CLAHE (p=0.01) des qu'albumentations
    # est importable : on enregistre ce qui s'est reellement applique.
    alb = etat_albumentations()
    print(f"  Ultralytics : bloc albumentations cache "
          f"{'ACTIF (Blur, MedianBlur, ToGray, CLAHE a p=0.01)' if alb else 'inactif'}")
    yml = build_fold_yaml(fold)
    npos, nneg = fold_counts(fold)
    batch, _ = cfg.batch_for("ultralytics")

    model = YOLO(weights)
    t0 = time.time()
    model.train(data=str(yml), epochs=cfg.EPOCHS, batch=batch, imgsz=cfg.IMGSZ,
                seed=cfg.SEED, verbose=False, val=True, patience=cfg.PATIENCE,
                project=PROJECT, name=f"{modele}_fold{fold}", exist_ok=True, **AUG)
    train_time = round(time.time() - t0, 1)

    best_epoch, epochs_run = _epochs_info(model.trainer)
    best_pt = Path(model.trainer.save_dir) / "weights" / "best.pt"
    if not best_pt.exists():
        raise FileNotFoundError("best.pt introuvable (entrainement interrompu ?)")
    saved = Path(cfg.SAVE_DIR) / f"{modele}_fold{fold}_best.pt"
    shutil.copy2(best_pt, saved)

    # Sauvegarde des hyperparametres reellement utilises (feuille "hyperparametres")
    args_src = Path(model.trainer.save_dir) / "args.yaml"
    if args_src.exists():
        shutil.copy2(args_src, Path(cfg.SAVE_DIR) / f"{modele}_fold{fold}_args.yaml")

    best = YOLO(str(saved))
    gt_json = cocoify.val_gt_json(fold)
    dt_json = Path(cfg.PRED_DIR) / f"{modele}_fold{fold}.json"
    predict_coco(best, fold_val_paths(fold), gt_json, dt_json)

    metrics = evaluate.evaluate_predictions(gt_json, dt_json)
    lat, lat_std = bench.latency_cpu_ms(best)
    stats = bench.model_stats(best, saved)

    return bench.base_row(modele, "ultralytics", fold, npos, nneg,
                          best_epoch=best_epoch, epochs_run=epochs_run,
                          train_time_s=train_time, latency_cpu_ms=round(lat, 3),
                          latency_std_ms=round(lat_std, 3),
                          notes=f"poids={weights} ; bloc albumentations "
                                f"{'actif' if alb else 'inactif'}",
                          **stats, **metrics)


def run_cv(models=None, folds=None):
    """Validation croisee des modeles Ultralytics du registre."""
    models = models or {n: s for n, (fw, s) in cfg.MODELS.items() if fw == "ultralytics"}
    out = None
    for modele, weights in models.items():
        out = bench.run_cv(modele, lambda f, m=modele, w=weights: run_fold(m, w, f),
                           folds=folds)
    return out
