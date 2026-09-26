# -*- coding: utf-8 -*-
"""RF-DETR Nano via le package `rfdetr` de Roboflow.

Dataset au format COCO "plat" (train / valid / test), construit a partir des
memes listes de tuiles que les autres modeles. Les hyperparametres restent ceux
par defaut de RF-DETR ; seules les augmentations (`augment.AUG_RFDETR`), les
epoques, le batch (8 x 4 = 32 effectif) et la resolution sont imposes.
"""

import shutil
import time
from pathlib import Path

from .. import augment, bench, cocoify, config as cfg, evaluate
from ..augment import AUG_RFDETR, patch_rfdetr_visibility
from ..folds import fold_val_paths
from . import arret_anticipe

MODELE = "RF-DETR-N"
# `predict` renvoie des class_id 0-based ; les categories COCO du benchmark sont
# 1-based (1 = Apterous, 2 = Alate).
CLASS_OFFSET = 1


def predict_coco(model, images, gt_json, out_json):
    """Exporte les detections du fold de validation au format COCO."""
    import json

    from PIL import Image

    gt = json.loads(Path(gt_json).read_text())
    name2id = {im["file_name"]: im["id"] for im in gt["images"]}
    by_name = {Path(p).name: Path(p) for p in images}

    dets = []
    for fn, iid in name2id.items():
        src = by_name.get(fn)
        if src is None or not src.exists():
            continue
        img = Image.open(src).convert("RGB")
        try:
            res = model.predict(img, threshold=cfg.CONF_EVAL)
        except TypeError:                       # signature plus ancienne
            res = model.predict(img)
        xyxy = getattr(res, "xyxy", None)
        if xyxy is None:
            continue
        conf = getattr(res, "confidence", None)
        cls = getattr(res, "class_id", None)
        for k in range(len(xyxy)):
            x1, y1, x2, y2 = (float(v) for v in xyxy[k])
            score = float(conf[k]) if conf is not None else 1.0
            if score < cfg.CONF_EVAL:
                continue
            cid = int(cls[k]) if cls is not None else 0
            dets.append({"image_id": iid, "category_id": cid + CLASS_OFFSET,
                         "bbox": [round(x1, 2), round(y1, 2),
                                  round(x2 - x1, 2), round(y2 - y1, 2)],
                         "score": round(score, 5)})
    return evaluate.write_detections(dets, out_json)


def run_fold(fold):
    """Entraine et evalue RF-DETR Nano sur un fold."""
    from rfdetr import RFDETRNano

    # Seuil de visibilite des boites tronquees par le recadrage interne du
    # framework (cf. docs/AUGMENTATION.md) : a poser avant la construction
    # du pipeline d'entrainement.
    seuil = patch_rfdetr_visibility()
    ds, npos, nneg = cocoify.build_coco_fold(fold, layout="flat")
    batch, accum = cfg.batch_for("rfdetr")
    out_dir = Path(cfg.WORK_ROOT) / "runs" / f"rfdetr_fold{fold}"
    out_dir.mkdir(parents=True, exist_ok=True)

    # RFDETRNanoConfig.resolution vaut 384 : sans cet argument, RF-DETR
    # s'entrainerait a une resolution inferieure aux six autres modeles.
    # 640 est bien divisible par patch_size * num_windows = 16 * 2.
    model = RFDETRNano(resolution=cfg.IMGSZ)
    kwargs = dict(dataset_dir=str(ds), epochs=cfg.MAX_EPOCHS, batch_size=batch,
                  grad_accum_steps=accum, output_dir=str(out_dir),
                  early_stopping=False,
                  aug_config=AUG_RFDETR)
    # Couches d'echelle natives de RF-DETR (scale_jitter : branche recadrage du
    # OneOf ; multi_scale et expanded_scales : resolution variable d'un lot a
    # l'autre). On les coupe pour que l'Affine bornee d'AUG_RFDETR soit la seule
    # source d'echelle, et on replie argument par argument si la version
    # installee n'en connait pas un.
    extras = dict(augment.RFDETR_COUPURES)
    if cfg.USE_WANDB:
        extras["wandb"] = True

    run = bench.wandb_run(MODELE, fold, {"batch": batch, "grad_accum": accum})
    t0 = time.time()
    while True:
        try:
            model.train(**kwargs, **extras)
            break
        except TypeError as e:
            refuse = next((k for k in list(extras) if k in str(e)), None)
            if refuse is None:
                raise                            # erreur sans rapport : on arrete
            extras.pop(refuse)
            print(f"  RF-DETR : argument '{refuse}' refuse par cette version "
                  f"({e}) -- nouvelle tentative sans lui")
    coupees = [k for k in augment.RFDETR_COUPURES if k in extras]
    manquantes = [k for k in augment.RFDETR_COUPURES if k not in extras]
    print(f"  RF-DETR : couches d'echelle coupees -> {', '.join(coupees) or 'aucune'}")
    if manquantes:
        print(f"  RF-DETR : ATTENTION, {', '.join(manquantes)} n'a pas pu etre "
              f"desactive : ce modele voit une variation d'echelle que les autres "
              f"n'ont pas")
    train_time = round(time.time() - t0, 1)

    # La branche Lightning de rfdetr journalise dans metrics.csv, pas dans le
    # log.txt JSON de l'ancien moteur : val/ema_mAP_50_95 par epoque.
    valeurs = arret_anticipe.lire_metrics_csv(out_dir / "metrics.csv")
    best_epoch = (max(range(len(valeurs)), key=lambda i: valeurs[i]) + 1
                  if valeurs else -1)
    epochs_run = len(valeurs) if valeurs else -1
    stopped_early = False          # budget fixe : plus d'arret anticipe

    # Poids retenus par RF-DETR (EMA si disponible). `BestModelCallback` est
    # ajoute inconditionnellement par le depot : le meilleur checkpoint existe
    # meme sans early stopping.
    saved = Path(cfg.SAVE_DIR) / f"{MODELE}_fold{fold}.pth"
    poids_evalues = None
    for name in ("checkpoint_best_ema.pth", "checkpoint_best_total.pth",
                 "checkpoint_best_regular.pth", "checkpoint.pth"):
        src = out_dir / name
        if src.exists():
            shutil.copy2(src, saved)
            poids_evalues = name
            break

    # `train()` se termine par `self.model.model = module.model` : l'objet en
    # memoire porte les poids de la DERNIERE epoque, pas le meilleur EMA. On
    # recharge donc le checkpoint sauvegarde, comme pour les autres frameworks.
    # Aucun repli : evaluer la derniere epoque produirait une ligne de resultat
    # silencieusement incomparable aux six autres modeles. bench.run_cv attrape
    # l'exception, marque le fold en erreur et passe au suivant.
    if not saved.exists():
        trouves = sorted(p.name for p in out_dir.glob("*.pth"))
        raise FileNotFoundError(
            f"Aucun checkpoint copie vers {saved}. Fichiers .pth presents dans "
            f"{out_dir} : {', '.join(trouves) or 'aucun'}. L'evaluation doit "
            f"porter sur le meilleur checkpoint EMA, pas sur la derniere epoque.")
    try:
        evalue = RFDETRNano.from_checkpoint(str(saved), trust_checkpoint=True)
    except Exception as e:
        raise RuntimeError(
            f"Rechargement de {saved} impossible : l'evaluation doit porter sur "
            f"le meilleur checkpoint EMA ({poids_evalues}), pas sur les poids de "
            f"la derniere epoque laisses en memoire par train().") from e
    print(f"  RF-DETR : evaluation sur {poids_evalues} (meilleur EMA)")

    gt_json = cocoify.val_gt_json(fold)
    dt_json = Path(cfg.PRED_DIR) / f"{MODELE}_fold{fold}.json"
    predict_coco(evalue, fold_val_paths(fold), gt_json, dt_json)

    metrics = evaluate.evaluate_predictions(gt_json, dt_json)
    lat, lat_std = bench.latency_cpu_ms(evalue)
    stats = bench.model_stats(evalue, saved if saved.exists() else None)

    row = bench.base_row(MODELE, "rfdetr", fold, npos, nneg,
                         best_epoch=best_epoch, epochs_run=epochs_run,
                         stopped_early=stopped_early,
                         train_time_s=train_time, latency_cpu_ms=round(lat, 3),
                         latency_std_ms=round(lat_std, 3),
                         notes=f"budget fixe {cfg.MAX_EPOCHS} epochs ; "
                               f"visibilite min "
                               f"{'non applicable' if seuil is None else f'{seuil:.0%}'} ; "
                               f"couches d'echelle coupees : "
                               f"{', '.join(coupees) or 'aucune'} ; "
                               f"poids evalues={poids_evalues}",
                         **stats, **metrics)
    bench.wandb_finish(run, metrics, suivi=row)
    return row


def run_cv(folds=None):
    """Validation croisee complete de RF-DETR Nano."""
    return bench.run_cv(MODELE, run_fold, folds=folds)
