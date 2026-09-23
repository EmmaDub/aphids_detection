# -*- coding: utf-8 -*-
"""YOLOX-Nano via le depot officiel Megvii-BaseDetection/YOLOX.

Le depot est clone puis installe en mode editable ; un fichier d'experience est
genere par fold a partir de `assets/yolox_exp_aphids.py` (les parametres du fold
sont passes par un JSON de meme nom). Les hyperparametres d'entrainement
(optimiseur SGD, LR cosinus, warmup, EMA, pertes) restent ceux de YOLOX ; seuls
le dataset, la resolution (640), le budget et l'augmentation sont imposes.
"""

import json
import shutil
import sys
import time
from pathlib import Path

from .. import bench, cocoify, config as cfg, evaluate, external
from ..augment import AUG_YOLOX
from ..folds import fold_val_paths
from . import arret_anticipe

ASSETS = Path(__file__).resolve().parent.parent / "assets"
MODELE = "YOLOX-Nano"
# COCODataset mappe les category_id tries sur 0..n-1 : 0 -> 1, 1 -> 2.
CLASS_OFFSET = 1


def setup(install=True):
    """Clone YOLOX et l'installe. Renvoie (dossier, commit)."""
    repo = external.clone("yolox")
    if install:
        req = repo / "requirements.txt"
        if req.exists():
            external.run([sys.executable, "-m", "pip", "install", "-q", "-r", str(req)],
                         check=False)
        external.run([sys.executable, "-m", "pip", "install", "-q", "-e", str(repo)],
                     check=False)
    return repo, external.git_commit(repo)


def write_exp(repo, fold, ds, out_dir):
    """Ecrit le fichier d'experience du fold (+ son JSON de parametres)."""
    exp_dir = Path(repo) / "exps" / "aphids"
    exp_dir.mkdir(parents=True, exist_ok=True)
    exp_py = exp_dir / f"yolox_nano_fold{fold}.py"
    shutil.copy2(ASSETS / "yolox_exp_aphids.py", exp_py)

    params = {
        "data_dir": str(ds),
        "train_ann": "instances_train2017.json",
        "val_ann": "instances_val2017.json",
        "num_classes": len(cfg.CLASS_NAMES),
        "num_workers": 2,
        "seed": cfg.SEED,
        "min_visibility": cfg.MIN_VISIBILITY,
        "imgsz": cfg.IMGSZ,
        "epochs": cfg.MAX_EPOCHS,
        "output_dir": str(out_dir),
        "exp_name": f"yolox_nano_fold{fold}",
        **{k: (list(v) if isinstance(v, tuple) else v) for k, v in AUG_YOLOX.items()},
    }
    exp_py.with_suffix(".json").write_text(json.dumps(params, indent=2))
    return exp_py


def train(repo, exp_py, ckpt, batch, run_dir):
    """Lance tools/train.py sous surveillance d'early stopping.

    YOLOX n'a pas d'arret anticipe : on lit le mAP50-95 que son evaluateur
    ecrit dans `train_log.txt` a chaque evaluation, et on met fin au processus
    quand la patience commune est epuisee. `best_ckpt.pth` est deja sauvegarde
    par YOLOX a chaque amelioration.

    Renvoie (batch reellement utilise, surveillant).
    """
    # tools/train.py de YOLOX n'a pas de flag --seed : la graine passe par l'Exp.
    def build(b):
        return [sys.executable, "tools/train.py", "-f", str(exp_py),
                "-d", "1", "-b", str(b), "--fp16", "-c", str(ckpt)]

    surveillant = arret_anticipe.Surveillant(
        cfg.EARLY_STOP_PATIENCE, cfg.EARLY_STOP_MIN_DELTA, nom=MODELE)
    surveillance = arret_anticipe.SurveillanceJournal(
        Path(run_dir) / "train_log.txt", arret_anticipe.lire_log_yolox, surveillant)

    _out, used = external.run_with_oom_retry(build, batch, cwd=repo,
                                             surveillance=surveillance)
    return used, surveillant


def predict_coco(model, exp, images, gt_json, out_json, device="cuda", batch=8):
    """Exporte les detections du fold de validation au format COCO."""
    import cv2
    import numpy as np
    import torch
    from yolox.data.data_augment import ValTransform
    from yolox.utils import postprocess

    preproc = ValTransform(legacy=False)
    test_size = exp.test_size
    gt = json.loads(Path(gt_json).read_text())
    name2id = {im["file_name"]: im["id"] for im in gt["images"]}
    by_name = {Path(p).name: Path(p) for p in images}
    files = [f for f in name2id if f in by_name]

    dets = []
    with torch.no_grad():
        for i in range(0, len(files), batch):
            chunk = files[i:i + batch]
            tensors, ratios = [], []
            for fn in chunk:
                img = cv2.imread(str(by_name[fn]))
                ratios.append(min(test_size[0] / img.shape[0], test_size[1] / img.shape[1]))
                arr, _ = preproc(img, None, test_size)
                tensors.append(torch.from_numpy(np.ascontiguousarray(arr)))
            x = torch.stack(tensors).float().to(device)
            outputs = model(x)
            outputs = postprocess(outputs, exp.num_classes, conf_thre=cfg.CONF_EVAL,
                                  nms_thre=cfg.NMS_IOU, class_agnostic=False)
            for k, fn in enumerate(chunk):
                out = outputs[k]
                if out is None:
                    continue
                iid, ratio = name2id[fn], ratios[k]
                for det in out.cpu().numpy():
                    x1, y1, x2, y2, obj_conf, cls_conf, cls_id = det[:7]
                    score = float(obj_conf) * float(cls_conf)
                    if score < cfg.CONF_EVAL:
                        continue
                    x1, y1, x2, y2 = (v / ratio for v in (x1, y1, x2, y2))
                    dets.append({"image_id": iid,
                                 "category_id": int(cls_id) + CLASS_OFFSET,
                                 "bbox": [round(float(x1), 2), round(float(y1), 2),
                                          round(float(x2 - x1), 2), round(float(y2 - y1), 2)],
                                 "score": round(score, 5)})
    return evaluate.write_detections(dets, out_json)


def run_fold(fold, repo=None, commit="", install=False):
    """Entraine et evalue YOLOX-Nano sur un fold."""
    import torch

    if repo is None:
        repo, commit = setup(install=install)
    external.add_to_path(repo)

    ds, npos, nneg = cocoify.build_coco_fold(fold, layout="coco")
    out_dir = Path(cfg.WORK_ROOT) / "runs" / "yolox"
    out_dir.mkdir(parents=True, exist_ok=True)
    exp_py = write_exp(repo, fold, ds, out_dir)
    ckpt0 = external.pretrained("yolox_nano")
    batch, _ = cfg.batch_for("yolox")

    run_dir = out_dir / f"yolox_nano_fold{fold}"
    run = bench.wandb_run(MODELE, fold, {"batch": batch})
    t0 = time.time()
    used_batch, surveillant = train(repo, exp_py, ckpt0, batch, run_dir)
    train_time = round(time.time() - t0, 1)
    best = run_dir / "best_ckpt.pth"
    if not best.exists():
        best = run_dir / "latest_ckpt.pth"
    if not best.exists():
        raise FileNotFoundError(f"Aucun checkpoint dans {run_dir}")
    saved = Path(cfg.SAVE_DIR) / f"{MODELE}_fold{fold}.pth"
    shutil.copy2(best, saved)

    state = torch.load(str(best), map_location="cpu", weights_only=False)
    best_epoch = int(state.get("start_epoch", -1))
    latest = run_dir / "latest_ckpt.pth"
    epochs_run = -1
    if latest.exists():
        epochs_run = int(torch.load(str(latest), map_location="cpu",
                                    weights_only=False).get("start_epoch", -1))

    from yolox.exp import get_exp
    exp = get_exp(str(exp_py), None)
    model = exp.get_model()
    model.load_state_dict(state["model"])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device).eval()

    gt_json = cocoify.val_gt_json(fold)
    dt_json = Path(cfg.PRED_DIR) / f"{MODELE}_fold{fold}.json"
    predict_coco(model, exp, fold_val_paths(fold), gt_json, dt_json, device=device)

    metrics = evaluate.evaluate_predictions(gt_json, dt_json)
    lat, lat_std = bench.latency_cpu_ms(model)
    stats = bench.model_stats(model, saved)

    row = bench.base_row(MODELE, "yolox", fold, npos, nneg,
                         best_epoch=best_epoch, epochs_run=epochs_run,
                         stopped_early=surveillant.declenche,
                         train_time_s=train_time, latency_cpu_ms=round(lat, 3),
                         latency_std_ms=round(lat_std, 3), batch=used_batch,
                         batch_effectif=used_batch, repo_commit=commit,
                         notes="flip vertical et gains HSV ajoutes par patch "
                               "(cf. assets/yolox_exp_aphids.py)",
                         **stats, **metrics)
    bench.wandb_finish(run, metrics, suivi=row)
    return row


def run_cv(folds=None, install=True):
    """Validation croisee complete de YOLOX-Nano."""
    repo, commit = setup(install=install)
    return bench.run_cv(MODELE,
                        lambda f: run_fold(f, repo=repo, commit=commit),
                        folds=folds)
