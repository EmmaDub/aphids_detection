# -*- coding: utf-8 -*-
"""Machinerie commune aux depots RT-DETR (lyuwenyu) et D-FINE (Peterande).

Les deux depots partagent la meme base de code (D-FINE est un fork de
RT-DETRv2) : meme systeme de configs YAML avec `__include__`, meme registre de
transforms, meme solver. Ce module :

  1. clone le depot officiel et y injecte `aphids_ops.py` (flip vertical +
     ColorJitter, cf. aphids_det/assets/aphids_ops.py) ;
  2. lit la liste de transforms de la config d'origine et la reecrit au plus
     proche de la reference d'augmentation, SANS toucher au reste des
     hyperparametres (optimiseur, LR, EMA, pertes restent ceux du depot) ;
  3. lance l'entrainement par le script officiel, avec reprise a batch reduit
     en cas d'OOM ;
  4. reconstruit le modele, exporte les predictions du fold de validation au
     format COCO et delegue les metriques a aphids_det.evaluate.

Choix des variantes
-------------------
`rtdetr_v2` est la variante par defaut pour RT-DETR-R18 : la version v1
(`rtdetr_pytorch`) importe `torchvision.datapoints`, supprime de torchvision
depuis la 0.17, et ne demarre donc pas sur un Colab recent. RT-DETRv2-R18
vient du meme depot officiel et utilise le meme backbone R18.
"""

import json
import os
import shutil
import sys
import time
from pathlib import Path

import yaml

from .. import bench, cocoify, config as cfg, evaluate, external
from ..augment import (DETR_AFFINE_OP, DETR_COLOR_OP, DETR_HFLIP_OP,
                       DETR_IOUCROP_OP, DETR_VFLIP_OP, DETR_ZOOMOUT_OP)
from ..folds import fold_val_paths

ASSETS = Path(__file__).resolve().parent.parent / "assets"

# ---------------------------------------------------------------- variantes
SPECS = {
    "rtdetr_v2": dict(
        modele="RT-DETR-R18", framework="rtdetr", repo="rtdetr",
        subdir="rtdetrv2_pytorch",
        base_cfg="configs/rtdetrv2/rtdetrv2_r18vd_120e_coco.yml",
        train_script="tools/train.py", amp_flag="--use-amp",
        epochs_key="epoches", weights="rtdetr_v2_r18",
        best_names=["best.pth"],
    ),
    "rtdetr_v1": dict(
        modele="RT-DETR-R18", framework="rtdetr", repo="rtdetr",
        subdir="rtdetr_pytorch",
        base_cfg="configs/rtdetr/rtdetr_r18vd_6x_coco.yml",
        train_script="tools/train.py", amp_flag="--amp",
        epochs_key="epoches", weights="rtdetr_v1_r18",
        best_names=[],            # v1 ne sauvegarde pas de "best" : on le choisit
    ),
    "dfine_n": dict(
        modele="D-FINE-N", framework="dfine", repo="dfine",
        subdir="",
        base_cfg="configs/dfine/dfine_hgnetv2_n_coco.yml",
        train_script="train.py", amp_flag="--use-amp",
        epochs_key="epochs", weights="dfine_n",
        best_names=["best_stg2.pth", "best_stg1.pth"],
    ),
}

DETR_NUM_WORKERS = 2          # Colab expose 2 vCPU
CLOSE_AUG_LAST_EPOCHS = 10    # equivalent de close_mosaic=10 d'Ultralytics


# -------------------------------------------------------------------- config
def _deep_merge(base, over):
    """Fusion recursive : les dicts sont fusionnes, le reste est remplace."""
    out = dict(base)
    for k, v in over.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def load_config_with_includes(path):
    """Charge un YAML du depot en resolvant `__include__` (chemins relatifs)."""
    path = Path(path)
    doc = yaml.safe_load(path.read_text()) or {}
    merged = {}
    for inc in doc.pop("__include__", []) or []:
        merged = _deep_merge(merged, load_config_with_includes((path.parent / inc).resolve()))
    return _deep_merge(merged, doc)


def patch_ops(ops, geom=None):
    """Reecrit la liste de transforms d'entrainement du depot.

    - `RandomPhotometricDistort` -> `ColorJitter` (memes amplitudes que hsv_s /
      hsv_v = 0.2, sans permutation de canaux) ;
    - `RandomHorizontalFlip` force a p = 0.5, suivi d'un `RandomVerticalFlip` ;
    - la geometrie suit `cfg.DETR_GEOM` : "reference" recale les bornes de
      `RandomZoomOut` et `RandomIoUCrop` sur `scale` et `translate` d'AUG,
      "affine" les remplace par un `RandomAffine` aux parametres d'Ultralytics,
      "natif" laisse les amplitudes du depot ;
    - le reste (Sanitize, Resize, conversions de fin) est conserve tel quel.

    Renvoie (nouvelles ops, correspondance ancien nom -> nouveau, None si l'op
    a disparu) ; la correspondance sert a reecrire la politique `stop_epoch`.
    """
    geom = cfg.DETR_GEOM if geom is None else geom
    new_ops, renamed = [], {}
    affine_pose = False
    for op in ops:
        t = op.get("type")
        if t == "RandomPhotometricDistort":
            new_ops.append(dict(DETR_COLOR_OP))
            renamed[t] = DETR_COLOR_OP["type"]
        elif t == "RandomHorizontalFlip":
            new_ops.append(dict(DETR_HFLIP_OP))
            new_ops.append(dict(DETR_VFLIP_OP))
        elif t in ("RandomZoomOut", "RandomIoUCrop"):
            if geom == "reference":
                new_ops.append(dict(DETR_ZOOMOUT_OP if t == "RandomZoomOut"
                                    else DETR_IOUCROP_OP))
            elif geom == "affine":
                if not affine_pose:        # une seule affine pour les deux ops
                    new_ops.append(dict(DETR_AFFINE_OP))
                    affine_pose = True
                    renamed[t] = DETR_AFFINE_OP["type"]
                else:
                    renamed[t] = None      # op retiree de la politique
            else:
                new_ops.append(dict(op))
        else:
            new_ops.append(dict(op))
    if geom == "affine" and not affine_pose:
        idx = next((i for i, o in enumerate(new_ops) if o.get("type") == "Resize"),
                   len(new_ops))
        new_ops.insert(idx, dict(DETR_AFFINE_OP))
    if not any(o.get("type") == DETR_VFLIP_OP["type"] for o in new_ops):
        # aucune HorizontalFlip dans la config d'origine : on insere les deux
        # miroirs juste avant le Resize final
        idx = next((i for i, o in enumerate(new_ops) if o.get("type") == "Resize"), len(new_ops))
        new_ops[idx:idx] = [dict(DETR_HFLIP_OP), dict(DETR_VFLIP_OP)]
    return new_ops, renamed


def write_config(variant, fold, ds, out_dir, batch, work):
    """Ecrit la config du fold dans le depot et renvoie son chemin.

    `work` est le dossier depuis lequel le script d'entrainement est lance
    (racine du depot, ou sous-dossier `rtdetrv2_pytorch` / `rtdetr_pytorch`).
    """
    spec = SPECS[variant]
    work = Path(work)
    base_path = work / spec["base_cfg"]
    base = load_config_with_includes(base_path)

    _, train_json, _, val_json = cocoify.coco_paths(ds, "coco")
    train_dir, _, val_dir, _ = cocoify.coco_paths(ds, "coco")

    train_dl = base.get("train_dataloader", {})
    batch_key = "total_batch_size" if "total_batch_size" in train_dl else "batch_size"
    ops = (train_dl.get("dataset", {}).get("transforms", {}) or {}).get("ops") or []
    new_ops, renamed = patch_ops(ops)

    ds_cfg = {
        "img_folder": str(train_dir), "ann_file": str(train_json),
        "return_masks": False,
        "transforms": {"type": "Compose", "ops": new_ops},
    }
    # Politique "stop_epoch" : laissee a sa valeur NATIVE. Seuls les noms d'ops
    # y sont mis a jour, puisque RandomPhotometricDistort a ete remplace.
    # ATTENTION : la valeur native (117 sur 120 epoques chez RT-DETRv2-R18, 148
    # sur 160 chez D-FINE-N) depasse le budget de 30 epoques du benchmark, donc
    # la coupure ne se declenche jamais -- ces deux modeles gardent leurs
    # augmentations jusqu'au bout, la ou YOLO coupe la mosaique sur les 10
    # dernieres epoques et YOLOX sur les 15 dernieres. Ecart a documenter.
    policy = (train_dl.get("dataset", {}).get("transforms", {}) or {}).get("policy")
    if isinstance(policy, dict):
        pol = dict(policy)
        # une op renommee suit son nouveau nom ; une op retiree sort de la politique
        pol["ops"] = [renamed.get(o, o) for o in pol.get("ops", [])
                      if renamed.get(o, o) is not None]
        ds_cfg["transforms"]["policy"] = pol
        print(f"  policy stop_epoch native : epoque {pol.get('epoch')} "
              f"(budget {cfg.EPOCHS} epoques"
              f"{' -- jamais atteinte' if (pol.get('epoch') or 0) >= cfg.EPOCHS else ''})")

    over = {
        "task": "detection",
        "num_classes": len(cfg.CLASS_NAMES) + 1,   # labels COCO 1-based -> max_id + 1
        "remap_mscoco_category": False,
        "output_dir": str(out_dir),
        "sync_bn": False,                          # mono-GPU
        spec["epochs_key"]: cfg.EPOCHS,
        "train_dataloader": {
            "dataset": ds_cfg,
            batch_key: batch,
            "shuffle": True, "drop_last": True, "num_workers": DETR_NUM_WORKERS,
        },
        "val_dataloader": {
            "dataset": {"img_folder": str(val_dir), "ann_file": str(val_json),
                        "return_masks": False},
            batch_key: max(1, batch // 2),
            "shuffle": False, "num_workers": DETR_NUM_WORKERS,
        },
    }
    if isinstance(train_dl.get("collate_fn"), dict):
        # collate_fn laisse natif lui aussi : son stop_epoch ne pilote que le
        # multi-echelle par lot, deja neutralise par les configs retenues
        # (scales: ~ chez RT-DETRv2-R18, base_size_repeat: ~ chez D-FINE-N).
        pass
    if variant == "rtdetr_v1":
        over["checkpoint_step"] = 1                # v1 ne sauvegarde pas de best.pth

    out_cfg = work / "configs" / "aphids" / f"{variant}_fold{fold}.yml"
    out_cfg.parent.mkdir(parents=True, exist_ok=True)
    doc = {"__include__": [os.path.relpath(base_path, out_cfg.parent).replace("\\", "/")]}
    doc.update(over)
    out_cfg.write_text(yaml.dump(doc, sort_keys=False, allow_unicode=True))
    print(f"  config ecrite : {out_cfg}  (batch {batch_key}={batch}, "
          f"{spec['epochs_key']}={cfg.EPOCHS})")
    return out_cfg


# --------------------------------------------------------------------- setup
def setup(variant, install=True):
    """Clone le depot, installe ses dependances et injecte les transforms."""
    spec = SPECS[variant]
    repo_root = external.clone(spec["repo"])
    work = repo_root / spec["subdir"] if spec["subdir"] else repo_root

    if install:
        req = work / "requirements.txt"
        if req.exists():
            external.run([sys.executable, "-m", "pip", "install", "-q", "-r", str(req)],
                         check=False)

    # Injection de aphids_ops.py (flip vertical + ColorJitter)
    target = work / "src" / "data" / "aphids_ops.py"
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(ASSETS / "aphids_ops.py", target)
    init = work / "src" / "data" / "__init__.py"
    line = "from .aphids_ops import *  # benchmark pucerons\n"
    if init.exists() and line not in init.read_text():
        with init.open("a") as f:
            f.write("\n" + line)
        print(f"  transforms du benchmark injectees dans {init}")

    return repo_root, work, external.git_commit(repo_root)


# ---------------------------------------------------------------- entrainement
def train(variant, work, cfg_path, out_dir, batch):
    """Lance le script d'entrainement officiel. Renvoie le batch reellement utilise."""
    spec = SPECS[variant]
    tuning = external.pretrained(spec["weights"])

    def build(b):
        if b != batch:            # relance apres OOM : config regeneree au bon batch
            _rewrite_batch(cfg_path, b)
        cmd = [sys.executable, spec["train_script"], "-c", str(cfg_path),
               "-t", str(tuning), spec["amp_flag"], "--seed", str(cfg.SEED)]
        # RT-DETR v1 ne connait pas --output-dir : output_dir est dans la config.
        if variant != "rtdetr_v1":
            cmd += ["--output-dir", str(out_dir)]
        return cmd

    _out, used = external.run_with_oom_retry(build, batch, cwd=work)
    return used


def _rewrite_batch(cfg_path, batch):
    doc = yaml.safe_load(Path(cfg_path).read_text())
    for section, div in (("train_dataloader", 1), ("val_dataloader", 2)):
        for key in ("total_batch_size", "batch_size"):
            if key in doc.get(section, {}):
                doc[section][key] = max(1, batch // div)
    Path(cfg_path).write_text(yaml.dump(doc, sort_keys=False, allow_unicode=True))


def read_log(out_dir):
    """(meilleure epoque, nb d'epoques faites) d'apres log.txt du depot."""
    log = Path(out_dir) / "log.txt"
    if not log.exists():
        return -1, -1
    best_ep, best_map, n = -1, -1.0, 0
    for line in log.read_text().splitlines():
        try:
            d = json.loads(line)
        except Exception:
            continue
        n += 1
        stats = d.get("test_coco_eval_bbox")
        if stats and float(stats[0]) > best_map:
            best_map, best_ep = float(stats[0]), int(d.get("epoch", n - 1))
    return (best_ep + 1 if best_ep >= 0 else -1), n      # epoques 1-based


def best_checkpoint(variant, out_dir, best_epoch, cleanup=True):
    """Chemin des poids retenus : `best*.pth` du depot, sinon l'epoque la plus forte."""
    out_dir = Path(out_dir)
    for name in SPECS[variant]["best_names"]:
        p = out_dir / name
        if p.exists():
            return p
    # RT-DETR v1 : checkpoint{epoch:04}.pth sauvegarde a chaque epoque
    if best_epoch and best_epoch > 0:
        p = out_dir / f"checkpoint{best_epoch - 1:04}.pth"
        if p.exists():
            if cleanup:
                for other in out_dir.glob("checkpoint0*.pth"):
                    if other != p:
                        other.unlink(missing_ok=True)
            return p
    last = out_dir / "checkpoint.pth"
    if last.exists():
        return last
    cands = sorted(out_dir.glob("*.pth"), key=lambda q: q.stat().st_mtime)
    if not cands:
        raise FileNotFoundError(f"Aucun checkpoint trouve dans {out_dir}")
    return cands[-1]


# ------------------------------------------------------------------ inference
def build_model(work, cfg_path, ckpt, device="cuda"):
    """Reconstruit (modele deploy, postprocesseur) a partir de la config et des poids."""
    import torch

    external.add_to_path(work)
    cwd = os.getcwd()
    os.chdir(work)
    try:
        from src.core import YAMLConfig
        conf = YAMLConfig(str(cfg_path), resume=str(ckpt))
        if "HGNetv2" in conf.yaml_cfg:
            conf.yaml_cfg["HGNetv2"]["pretrained"] = False    # inutile a l'inference
        state = torch.load(str(ckpt), map_location="cpu", weights_only=False)
        if "ema" in state and state["ema"] is not None:
            weights = state["ema"]["module"]
        else:
            weights = state["model"]
        conf.model.load_state_dict(weights)
        model = conf.model.deploy().to(device).eval()
        post = conf.postprocessor.deploy().to(device).eval()
        return model, post
    finally:
        os.chdir(cwd)


def predict_coco(model, post, images, gt_json, out_json, device="cuda", batch=8):
    """Exporte les detections du fold de validation au format COCO.

    `category_id` = label predit : avec `remap_mscoco_category: False` et
    `num_classes = 3`, le depot apprend directement les identifiants COCO 1-based
    du dataset (1 = Apterous, 2 = Alate).
    """
    import numpy as np
    import torch
    from PIL import Image

    gt = json.loads(Path(gt_json).read_text())
    name2id = {im["file_name"]: im["id"] for im in gt["images"]}
    images = {Path(p).name: Path(p) for p in images}
    files = [f for f in name2id if f in images]

    dets = []
    size = torch.tensor([[cfg.IMGSZ, cfg.IMGSZ]], device=device)
    with torch.no_grad():
        for i in range(0, len(files), batch):
            chunk = files[i:i + batch]
            tensors = []
            for fn in chunk:
                im = Image.open(images[fn]).convert("RGB").resize((cfg.IMGSZ, cfg.IMGSZ))
                # equivalent de ConvertPILImage(dtype=float32, scale=True) du depot
                arr = np.asarray(im, dtype=np.float32) / 255.0
                tensors.append(torch.from_numpy(arr).permute(2, 0, 1))
            x = torch.stack(tensors).to(device)
            labels, boxes, scores = post(model(x), size.repeat(len(chunk), 1))
            for k, fn in enumerate(chunk):
                iid = name2id[fn]
                for lab, box, sc in zip(labels[k].tolist(), boxes[k].tolist(),
                                        scores[k].tolist()):
                    if sc < cfg.CONF_EVAL or int(lab) <= 0:
                        continue
                    x1, y1, x2, y2 = box
                    dets.append({"image_id": iid, "category_id": int(lab),
                                 "bbox": [round(x1, 2), round(y1, 2),
                                          round(x2 - x1, 2), round(y2 - y1, 2)],
                                 "score": round(float(sc), 5)})
    return evaluate.write_detections(dets, out_json)


# ------------------------------------------------------------------ pipeline
def run_fold(variant, fold, work=None, commit="", install=False):
    """Entraine et evalue une variante DETR sur un fold. Renvoie la ligne de resultat."""
    spec = SPECS[variant]
    modele = spec["modele"]
    if work is None:
        _root, work, commit = setup(variant, install=install)

    ds, npos, nneg = cocoify.build_coco_fold(fold, layout="coco")
    out_dir = Path(cfg.WORK_ROOT) / "runs" / f"{variant}_fold{fold}"
    out_dir.mkdir(parents=True, exist_ok=True)
    batch, _accum = cfg.batch_for(spec["framework"])

    cfg_path = write_config(variant, fold, ds, out_dir, batch, work)
    run = bench.wandb_run(modele, fold, {"variante": variant, "batch": batch})

    t0 = time.time()
    used_batch = train(variant, work, cfg_path, out_dir, batch)
    train_time = round(time.time() - t0, 1)

    best_epoch, epochs_run = read_log(out_dir)
    ckpt = best_checkpoint(variant, out_dir, best_epoch)
    saved = Path(cfg.SAVE_DIR) / f"{modele}_fold{fold}{ckpt.suffix}"
    shutil.copy2(ckpt, saved)

    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, post = build_model(work, cfg_path, ckpt, device=device)

    gt_json = cocoify.val_gt_json(fold)
    dt_json = Path(cfg.PRED_DIR) / f"{modele}_fold{fold}.json"
    predict_coco(model, post, fold_val_paths(fold), gt_json, dt_json, device=device)

    metrics = evaluate.evaluate_predictions(gt_json, dt_json)
    lat, lat_std = bench.latency_cpu_ms(model)
    stats = bench.model_stats(model, saved)

    row = bench.base_row(modele, spec["framework"], fold, npos, nneg,
                         best_epoch=best_epoch, epochs_run=epochs_run,
                         train_time_s=train_time, latency_cpu_ms=round(lat, 3),
                         latency_std_ms=round(lat_std, 3), batch=used_batch,
                         batch_effectif=used_batch, repo_commit=commit,
                         notes=f"variante={variant}; poids={spec['weights']}",
                         **stats, **metrics)
    bench.wandb_finish(run, metrics)
    return row


def run_cv(variant, folds=None, install=True):
    """Boucle de validation croisee complete pour une variante DETR."""
    _root, work, commit = setup(variant, install=install)
    spec = SPECS[variant]
    return bench.run_cv(spec["modele"],
                        lambda f: run_fold(variant, f, work=work, commit=commit),
                        folds=folds)
