# -*- coding: utf-8 -*-
"""Test hors-ligne de la chaine folds -> COCO -> evaluation unifiee.

Ne demande ni GPU, ni donnees, ni framework de detection : seulement pandas,
numpy, pyyaml et pycocotools. A lancer avant une campagne pour verifier que la
plomberie (listes figees, conversion COCO, metriques, CSV) est intacte :

    python tests/test_pipeline.py
"""

import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import aphids_det.config as cfg

tmp = Path(tempfile.mkdtemp())
cfg.WORK_ROOT = tmp / "work"
cfg.OUT_DIR = tmp / "out"
cfg.refresh()

from aphids_det import bench, cocoify, evaluate, folds
from aphids_det.runners import detr_repo

# --- fixture : fold_0 avec 2 tuiles pucerons + 1 fond, fold_1 pour le train ---
root = Path(cfg.FOLDS_ROOT)
for f in (0, 1):
    (root / f"fold_{f}" / "images").mkdir(parents=True, exist_ok=True)
    (root / f"fold_{f}" / "labels").mkdir(parents=True, exist_ok=True)

# fold 0 (validation) : t0 a 2 boites (classes 0 et 1), t1 a 1 boite, bg0 rien
(root / "fold_0" / "images" / "t0.jpg").write_bytes(b"x")
(root / "fold_0" / "images" / "t1.jpg").write_bytes(b"x")
(root / "fold_0" / "images" / "bg0.jpg").write_bytes(b"x")
(root / "fold_0" / "labels" / "t0.txt").write_text(
    "0 0.5 0.5 0.1 0.1\n1 0.25 0.25 0.05 0.05\n")
(root / "fold_0" / "labels" / "t1.txt").write_text("0 0.75 0.75 0.2 0.2\n")

# fold 1 (train) : 1 puceron + 10 fonds -> neg_ratio 3 doit en garder 3
(root / "fold_1" / "images" / "p0.jpg").write_bytes(b"x")
(root / "fold_1" / "labels" / "p0.txt").write_text("0 0.5 0.5 0.1 0.1\n")
for i in range(10):
    (root / "fold_1" / "images" / f"b{i}.jpg").write_bytes(b"x")

cfg.N_CV_FOLDS = 2
cfg.CV_FOLDS = [0, 1]

# --- 1. liste de train figee + sous-echantillonnage des fonds ---
npos, nneg = folds.fold_counts(0)
print(f"[train fold0] pucerons={npos} fonds={nneg}  (attendu 1 / 3)")
assert (npos, nneg) == (1, 3), (npos, nneg)
frozen = Path(cfg.SPLITS_FROZEN) / "train_fold0_neg3.txt"
assert frozen.exists(), "la liste de train doit etre figee sur le Drive"
assert folds.fold_train_paths(0) == [l for l in frozen.read_text().splitlines() if l]

# --- 2. ground-truth COCO du fold de validation ---
gt_path = cocoify.val_gt_json(0)
gt = json.loads(gt_path.read_text())
print(f"[GT fold0] images={len(gt['images'])} annotations={len(gt['annotations'])} "
      f"categories={[c['name'] for c in gt['categories']]}")
assert len(gt["images"]) == 3 and len(gt["annotations"]) == 3
# boite 0 : centre 0.5,0.5 taille 0.1 -> x=288 y=288 w=64 h=64 en 640 px
a0 = next(a for a in gt["annotations"] if a["category_id"] == 1)
assert a0["bbox"] == [288.0, 288.0, 64.0, 64.0], a0["bbox"]

# --- 3. predictions parfaites -> metriques maximales ---
perfect = [{"image_id": a["image_id"], "category_id": a["category_id"],
            "bbox": a["bbox"], "score": 0.9} for a in gt["annotations"]]
dt = Path(cfg.PRED_DIR) / "perfect.json"
evaluate.write_detections(perfect, dt)
m = evaluate.evaluate_predictions(gt_path, dt)
print(f"[parfait] map50={m['map50_macro']} map50-95={m['map5095_macro']} "
      f"F1_apt={m['Apterous_aphid_F1']} F1_ala={m['Alate_aphid_F1']} "
      f"TP={m['Apterous_aphid_TP']}/{m['Alate_aphid_TP']}")
assert m["map50_macro"] == 1.0 and m["map5095_macro"] == 1.0
assert m["Apterous_aphid_F1"] == 1.0 and m["Alate_aphid_F1"] == 1.0
assert m["Apterous_aphid_TP"] == 2 and m["Alate_aphid_TP"] == 1

# --- 4. predictions decalees / manquantes -> FP et FN ---
shifted = [{"image_id": perfect[0]["image_id"], "category_id": 1,
            "bbox": [0.0, 0.0, 64.0, 64.0], "score": 0.9}]
dt2 = Path(cfg.PRED_DIR) / "shifted.json"
evaluate.write_detections(shifted, dt2)
m2 = evaluate.evaluate_predictions(gt_path, dt2)
print(f"[decale] TP={m2['Apterous_aphid_TP']} FP={m2['Apterous_aphid_FP']} "
      f"FN={m2['Apterous_aphid_FN']} / Alate FN={m2['Alate_aphid_FN']}")
assert m2["Apterous_aphid_TP"] == 0 and m2["Apterous_aphid_FP"] == 1
assert m2["Apterous_aphid_FN"] == 2 and m2["Alate_aphid_FN"] == 1

# --- 5. detections vides -> zeros, pas d'exception ---
dt3 = Path(cfg.PRED_DIR) / "empty.json"
evaluate.write_detections([], dt3)
m3 = evaluate.evaluate_predictions(gt_path, dt3)
assert m3["map50_macro"] == 0.0 and m3["Apterous_aphid_FN"] == 2
print("[vide] ok")

# --- 6. seuil de confiance P/R (conf < 0.25 ignore) ---
lowconf = [dict(p, score=0.1) for p in perfect]
dt4 = Path(cfg.PRED_DIR) / "lowconf.json"
evaluate.write_detections(lowconf, dt4)
m4 = evaluate.evaluate_predictions(gt_path, dt4)
print(f"[conf 0.1] map50={m4['map50_macro']} (compte) TP={m4['Apterous_aphid_TP']} (ignore)")
assert m4["map50_macro"] == 1.0 and m4["Apterous_aphid_TP"] == 0

# --- 7. dataset COCO "coco" (RT-DETR / D-FINE / YOLOX) ---
ds, p, n = cocoify.build_coco_fold(0, layout="coco", verbose=False)
tr_dir, tr_json, va_dir, va_json = cocoify.coco_paths(ds, "coco")
print(f"[layout coco] {tr_json.relative_to(ds)} / {va_json.relative_to(ds)} existent="
      f"{tr_json.exists()},{va_json.exists()}")
assert tr_json.exists() and va_json.exists()
assert json.loads(va_json.read_text())["annotations"] == gt["annotations"]

# --- 8. layout "flat" (RF-DETR) : meme GT que le fichier de reference ---
ds2, _, _ = cocoify.build_coco_fold(0, layout="flat", verbose=False)
_, _, _, va_json2 = cocoify.coco_paths(ds2, "flat")
assert json.loads(va_json2.read_text())["images"] == gt["images"]
print("[layout flat] GT identique au fichier de reference -> comparaison valide")

# --- 9. reecriture des transforms RT-DETRv2 ---
ops_v2 = [{"type": "RandomPhotometricDistort", "p": 0.5},
          {"type": "RandomZoomOut", "fill": 0},
          {"type": "RandomIoUCrop", "p": 0.8},
          {"type": "SanitizeBoundingBoxes", "min_size": 1},
          {"type": "RandomHorizontalFlip"},
          {"type": "Resize", "size": [640, 640]},
          {"type": "SanitizeBoundingBoxes", "min_size": 1},
          {"type": "ConvertPILImage", "dtype": "float32", "scale": True},
          {"type": "ConvertBoxes", "fmt": "cxcywh", "normalize": True}]
new_ops, renamed = detr_repo.patch_ops(ops_v2)
types = [o["type"] for o in new_ops]
print("[ops RT-DETR]", types)
assert "RandomPhotometricDistort" not in types
assert types.index("ColorJitter") == 0
assert types[types.index("RandomHorizontalFlip") + 1] == "RandomVerticalFlip"
assert types[-2:] == ["ConvertPILImage", "ConvertBoxes"], "queue du pipeline preservee"
assert renamed == {"RandomPhotometricDistort": "ColorJitter"}

# --- 10. CSV de resultats : ordre des colonnes et reprise ---
row = bench.base_row("TEST", "ultralytics", 0, npos, nneg, **m)
bench.save_rows([row])
rows, done = bench.load_rows()
print(f"[CSV] {len(rows)} ligne(s), done={done}")
assert done == {("TEST", 0)}
import pandas as pd
cols = list(pd.read_csv(cfg.CSV_CV).columns)
assert cols[:6] == ["date", "modele", "framework", "fold", "map50_macro", "map5095_macro"]

# --- 11. geometrie des DETR : les trois modes ---
for mode, attendu in [("reference", {"RandomZoomOut", "RandomIoUCrop"}),
                      ("affine", {"RandomAffine"}),
                      ("natif", {"RandomZoomOut", "RandomIoUCrop"})]:
    ops_mode, _ren = detr_repo.patch_ops(ops_v2, geom=mode)
    noms = {o["type"] for o in ops_mode}
    assert attendu <= noms, (mode, noms)
    geo = {o["type"]: o for o in ops_mode}
    if mode == "reference":
        # bornes calees sur scale=0.25 : objet dans x[0.75, 1.25]
        assert geo["RandomZoomOut"]["side_range"] == [1.0, 1.333], geo["RandomZoomOut"]
        assert geo["RandomZoomOut"]["fill"] == 114, "remplissage gris comme YOLO"
        assert geo["RandomIoUCrop"]["min_scale"] == 0.8, geo["RandomIoUCrop"]
        assert geo["RandomIoUCrop"]["max_aspect_ratio"] == 1.1, "aspect quasi isotrope"
    if mode == "affine":
        assert geo["RandomAffine"]["scale"] == [0.75, 1.25], geo["RandomAffine"]
        assert geo["RandomAffine"]["translate"] == [0.1, 0.1]
        assert "RandomIoUCrop" not in noms and "RandomZoomOut" not in noms
    if mode == "natif":
        assert geo["RandomZoomOut"] == {"type": "RandomZoomOut", "fill": 0}
print("[geometrie DETR] reference / affine / natif : bornes conformes")

# --- 12. seuil de visibilite des boites tronquees ---
import numpy as np

from aphids_det.augment import masque_visibilite

aires = np.array([100.0, 100.0, 100.0])
apres = np.array([[0, 0, 10, 10],      # 100 % visible -> gardee
                  [0, 0, 10, 3],       #  30 % -> gardee (seuil 20 %)
                  [0, 0, 10, 1]])      #  10 % -> retiree
m = masque_visibilite(aires, apres, min_visibility=0.20)
print(f"[visibilite 20%] gardees = {m.tolist()}")
assert m.tolist() == [True, True, False], m
assert masque_visibilite(aires, apres, min_visibility=0.5).tolist() == [True, False, False]

# --- 13. arret anticipe : patience, min_delta, direction ---
from aphids_det.runners import arret_anticipe

surv = arret_anticipe.Surveillant(patience=3, min_delta=0.001, nom="test")
assert not surv.observer([0.10, 0.20, 0.30, 0.40])      # progression : pas d'arret
assert (surv.meilleure_epoque, surv.vues) == (4, 4)
# plateau : arret apres 3 epoques sans gain superieur a min_delta
assert surv.observer([0.10, 0.20, 0.30, 0.40, 0.4005, 0.4002, 0.4001])
assert surv.resume() == {"best_epoch": 4, "epochs_run": 7,
                         "stopped_early": True, "meilleure": 0.40}
print(f"[arret anticipe] {surv.resume()}")

# un gain inferieur a min_delta ne remet pas le compteur a zero
surv2 = arret_anticipe.Surveillant(patience=2, min_delta=0.01, nom="test2")
assert surv2.observer([0.50, 0.505, 0.509])
assert surv2.meilleure_epoque == 1

# lecteurs de journaux : DETR (JSON par ligne) et YOLOX (sortie COCO)
log = Path(cfg.WORK_ROOT) / "log.txt"
log.write_text('{"epoch": 0, "test_coco_eval_bbox": [0.11, 0.2]}\n'
               '{"epoch": 1, "test_coco_eval_bbox": [0.33, 0.4]}\n')
assert arret_anticipe.lire_log_json(log) == [0.11, 0.33]
ylog = Path(cfg.WORK_ROOT) / "train_log.txt"
# Sortie reelle de summarize() de pycocotools : 12 lignes par evaluation, dont
# 6 "Average Recall" qui partagent la plage IoU=0.50:0.95 et l'aire "all".
# Une seule valeur doit en sortir, sinon la patience serait divisee par 4.
SUMMARIZE = """ Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = {ap}
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.750
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.564
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.270
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.526
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = -1.000
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.384
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.529
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.529
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.367
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.577
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = -1.000
"""
ylog.write_text(SUMMARIZE.format(ap="0.123") + SUMMARIZE.format(ap="0.456"))
lues = arret_anticipe.lire_log_yolox(ylog)
assert lues == [0.123, 0.456], lues       # une valeur par evaluation, pas quatre
print(f"[lecteurs de journaux] DETR : OK | YOLOX : {len(lues)} valeurs pour "
      f"2 evaluations de 12 lignes")

# --- 14. tracabilite : stopped_early dans le CSV ---
assert "stopped_early" in bench.result_columns()
suivi = bench.base_row("TEST2", "yolox", 1, 10, 5, stopped_early=True, epochs_run=42)
assert suivi["stopped_early"] is True and suivi["epochs_budget"] == cfg.MAX_EPOCHS
assert bench.base_row("TEST3", "yolox", 2, 1, 1)["stopped_early"] is None
print(f"[tracabilite] stopped_early present | plafond {cfg.MAX_EPOCHS} | "
      f"patience {cfg.EARLY_STOP_PATIENCE} | min_delta {cfg.EARLY_STOP_MIN_DELTA}")

print("\nTOUS LES TESTS PASSENT")
