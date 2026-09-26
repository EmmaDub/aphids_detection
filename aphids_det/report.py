# -*- coding: utf-8 -*-
"""Synthese du benchmark : moyennes par modele, hyperparametres, export Excel.

A lancer une fois les 7 modeles passes (notebooks 01 a 05). Tout part du CSV
unique `cfg.CSV_CV`, alimente par les runners.
"""

import datetime
from pathlib import Path

import pandas as pd
import yaml

from . import augment, config as cfg

# Hyperparametres non modifies par le benchmark : ce qui suit documente ce que
# chaque depot applique PAR DEFAUT, pour la feuille "hyperparametres".
HP_DEFAULTS = [
    dict(modele="YOLO26n", framework="ultralytics", poids_init="yolo26n.pt (COCO)",
         optimiseur="auto (Ultralytics)", lr="defaut Ultralytics",
         selection="best.pt, fitness Ultralytics (0.9 mAP50-95 + 0.1 mAP50)", early_stopping=f"aucun (budget fixe {cfg.MAX_EPOCHS} epochs)"),
    dict(modele="YOLO11n", framework="ultralytics", poids_init="yolo11n.pt (COCO)",
         optimiseur="auto (Ultralytics)", lr="defaut Ultralytics",
         selection="best.pt, fitness Ultralytics (0.9 mAP50-95 + 0.1 mAP50)", early_stopping=f"aucun (budget fixe {cfg.MAX_EPOCHS} epochs)"),
    dict(modele="YOLO12n", framework="ultralytics", poids_init="yolo12n.pt (COCO)",
         optimiseur="auto (Ultralytics)", lr="defaut Ultralytics",
         selection="best.pt, fitness Ultralytics (0.9 mAP50-95 + 0.1 mAP50)", early_stopping=f"aucun (budget fixe {cfg.MAX_EPOCHS} epochs)"),
    dict(modele="RF-DETR-N", framework="rfdetr", poids_init="RFDETRNano (pre-entraine)",
         optimiseur="AdamW (defaut RF-DETR)", lr="defaut RF-DETR",
         selection="checkpoint_best_ema.pth (meilleur AP50-95 COCO natif, EMA)",
         early_stopping=f"aucun (budget fixe {cfg.MAX_EPOCHS} epochs)"),
    dict(modele="RT-DETR-R18", framework="rtdetr",
         poids_init="rtdetrv2_r18vd_120e_coco.pth (COCO)",
         optimiseur="AdamW (defaut depot)", lr="defaut depot",
         selection="best.pth (meilleur AP50-95 COCO natif)",
         early_stopping=f"aucun (budget fixe {cfg.MAX_EPOCHS} epochs)"),
    dict(modele="D-FINE-N", framework="dfine", poids_init="dfine_n_coco.pth (COCO)",
         optimiseur="AdamW (defaut depot)", lr="defaut depot",
         selection="best_stg1.pth (meilleur AP50-95 COCO natif)",
         early_stopping=f"aucun (budget fixe {cfg.MAX_EPOCHS} epochs)"),
    dict(modele="YOLOX-Nano", framework="yolox", poids_init="yolox_nano.pth (COCO)",
         optimiseur="SGD + cosinus (defaut YOLOX)", lr="defaut YOLOX",
         selection="best_ckpt.pth (meilleur AP50-95 COCO natif)",
         early_stopping=f"aucun (budget fixe {cfg.MAX_EPOCHS} epochs)"),
]

ULTRALYTICS_HP_KEYS = ["optimizer", "lr0", "lrf", "momentum", "weight_decay",
                       "warmup_epochs", "box", "cls", "dfl", "close_mosaic", "cos_lr", "amp"]


def hyperparameters_table():
    """Feuille "hyperparametres" : protocole commun + defauts de chaque depot."""
    rows = []
    for hp in HP_DEFAULTS:
        row = dict(hp)
        batch, accum = cfg.batch_for(hp["framework"])
        row.update({
            "epochs_budget": cfg.MAX_EPOCHS, "imgsz": cfg.IMGSZ,
            "batch": batch, "grad_accum": accum, "batch_effectif": batch * accum,
            "neg_ratio": cfg.NEG_RATIO, "seed": cfg.SEED,
            "validation": "fold complet (positifs + fonds)",
            "evaluation": f"COCO unifiee, conf {cfg.CONF_EVAL}, maxDets {cfg.MAX_DET}",
        })
        # Valeurs reellement utilisees par Ultralytics (args.yaml du fold 0)
        if hp["framework"] == "ultralytics":
            args = Path(cfg.SAVE_DIR) / f"{hp['modele']}_fold0_args.yaml"
            if args.exists():
                a = yaml.safe_load(args.read_text())
                row.update({k: a.get(k) for k in ULTRALYTICS_HP_KEYS})
        rows.append(row)
    return pd.DataFrame(rows)


def summary(df=None, ecarts=True):
    """Moyennes par modele sur les folds, triees par mAP50.

    Avec `ecarts=True`, chaque metrique de qualite recoit une colonne
    `<metrique>_std` : l'ecart-type entre folds, indispensable pour savoir si un
    ecart entre deux modeles depasse la variabilite de la validation croisee.
    """
    df = pd.read_csv(cfg.CSV_CV) if df is None else df
    num = [c for c in df.columns
           if c not in ("modele", "fold", "framework", "date", "repo_commit",
                        "versions", "notes")
           and pd.api.types.is_numeric_dtype(df[c])]
    moy = df.groupby("modele")[num].mean().round(4).reset_index()

    if ecarts:
        classes = [c for c in num
                   if c.startswith(tuple(cfg.CLASS_NAMES.values()))]
        cibles = [c for c in ("map50_macro", "map5095_macro", *classes) if c in num]
        std = (df.groupby("modele")[cibles].std(ddof=1).round(4)
               .rename(columns={c: f"{c}_std" for c in cibles}).reset_index())
        moy = moy.merge(std, on="modele")
        # chaque ecart-type juste apres sa moyenne
        ordre = ["modele"]
        for c in moy.columns:
            if c in ("modele",) or c.endswith("_std"):
                continue
            ordre.append(c)
            if f"{c}_std" in moy.columns:
                ordre.append(f"{c}_std")
        moy = moy[ordre]

    counts = df.groupby("modele")["fold"].count().rename("n_folds").reset_index()
    moy = moy.merge(counts, on="modele")
    return moy.sort_values("map50_macro", ascending=False).reset_index(drop=True)


def to_excel(path=None):
    """Ecrit le classeur de synthese (4 feuilles) et renvoie son chemin."""
    df = pd.read_csv(cfg.CSV_CV)
    stamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M")
    path = Path(path or (Path(cfg.OUT_DIR) / f"benchmark_detection_{stamp}.xlsx"))

    moy = summary(df)
    with pd.ExcelWriter(path, engine="openpyxl") as w:
        hyperparameters_table().to_excel(w, sheet_name="hyperparametres", index=False)
        augment.table().to_excel(w, sheet_name="augmentations", index=False)
        df.to_excel(w, sheet_name="resultats_par_fold", index=False)
        moy.to_excel(w, sheet_name="moyennes_par_modele", index=False)

    print("Excel ecrit ->", path)
    cols = ["modele", "n_folds", "map50_macro", "map50_macro_std",
            "map5095_macro", "map5095_macro_std",
            "latency_cpu_ms", "n_params_M", "size_MB", "train_time_s"]
    print("\n=== Moyennes par modele (triees par mAP50) ===")
    print(moy[[c for c in cols if c in moy.columns]].to_string(index=False))
    return path
