# -*- coding: utf-8 -*-
"""Harnais commun : latence CPU, taille du modele, CSV de resultats, boucle CV.

Toutes les colonnes "cout" (`train_time_s`, `latency_cpu_ms`, `latency_std_ms`,
`n_params_M`, `size_MB`) sont mesurees ici, de la meme facon pour tous les
frameworks, afin qu'elles restent comparables :

  - `latency_cpu_ms` : passe avant d'un batch de 1 image `IMGSZ x IMGSZ` sur CPU,
    directement sur le `nn.Module` (sans pre/post-traitement ni NMS) ;
  - `n_params_M` / `size_MB` : parametres + buffers du module, en float32 ;
  - `ckpt_MB` : taille reelle du fichier de poids sauvegarde.
"""

import datetime
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd

from . import config as cfg


# ------------------------------------------------------------- module PyTorch
def torch_module(obj, max_depth=5):
    """Remonte la chaine `.model` jusqu'au premier `nn.Module` trouve.

    Gere YOLO(...).model (Ultralytics), RFDETRNano().model.model (RF-DETR),
    et les modeles construits directement (RT-DETR, D-FINE, YOLOX).
    """
    import torch.nn as nn

    for _ in range(max_depth):
        if isinstance(obj, nn.Module):
            return obj
        if hasattr(obj, "model"):
            obj = obj.model
        else:
            break
    return obj if isinstance(obj, nn.Module) else None


def model_stats(obj, ckpt_path=None):
    """n_params_M, size_MB (poids float32) et ckpt_MB (fichier sur disque)."""
    out = {"n_params_M": float("nan"), "size_MB": float("nan"), "ckpt_MB": float("nan")}
    module = torch_module(obj)
    if module is not None:
        tensors = list(module.parameters()) + list(module.buffers())
        n_params = sum(p.numel() for p in module.parameters())
        n_bytes = sum(t.numel() * t.element_size() for t in tensors)
        out["n_params_M"] = round(n_params / 1e6, 3)
        out["size_MB"] = round(n_bytes / 1e6, 2)
    if ckpt_path and Path(ckpt_path).exists():
        out["ckpt_MB"] = round(Path(ckpt_path).stat().st_size / 1e6, 2)
    return out


def latency_cpu_ms(obj, imgsz=None, warmup=None, iters=None):
    """Latence CPU par image (forward, batch=1) : moyenne et ecart-type en ms."""
    import torch

    module = torch_module(obj)
    if module is None:
        print("  latence CPU non mesurable (module PyTorch introuvable)")
        return float("nan"), float("nan")

    imgsz = imgsz or cfg.IMGSZ
    warmup = cfg.LATENCY_WARMUP if warmup is None else warmup
    iters = cfg.LATENCY_ITERS if iters is None else iters

    was_training = module.training
    device_before = next(module.parameters()).device
    module = module.to("cpu").eval()
    x = torch.randn(1, 3, imgsz, imgsz)
    try:
        with torch.no_grad():
            for _ in range(warmup):
                module(x)
            ts = []
            for _ in range(iters):
                t0 = time.perf_counter()
                module(x)
                ts.append((time.perf_counter() - t0) * 1000)
        return float(np.mean(ts)), float(np.std(ts))
    except Exception as e:
        print(f"  latence CPU non mesuree : {type(e).__name__}: {e}")
        return float("nan"), float("nan")
    finally:
        module.to(device_before)
        module.train(was_training)


# ------------------------------------------------------------------ resultats
def _class_columns():
    names = [cfg.CLASS_NAMES[i] for i in range(len(cfg.CLASS_NAMES))]
    cols = []
    for nm in names:
        cols += [f"{nm}_map50", f"{nm}_map5095", f"{nm}_P", f"{nm}_R", f"{nm}_F1",
                 f"{nm}_TP", f"{nm}_FP", f"{nm}_FN"]
    return cols


def result_columns():
    """Ordre de colonnes du CSV de resultats."""
    return (["date", "modele", "framework", "fold", "map50_macro", "map5095_macro"]
            + _class_columns()
            + ["best_epoch", "epochs_run", "stopped_early", "epochs_budget",
               "train_time_s",
               "latency_cpu_ms", "latency_std_ms", "n_params_M", "size_MB", "ckpt_MB",
               "batch", "grad_accum", "batch_effectif", "imgsz", "neg_ratio",
               "n_pucerons_train", "n_fonds_train", "repo_commit", "versions",
               "notes"])


def load_rows(csv_path=None):
    """(lignes existantes, ensemble des (modele, fold) deja faits)."""
    csv_path = Path(csv_path or cfg.CSV_CV)
    if not csv_path.exists():
        return [], set()
    df = pd.read_csv(csv_path)
    rows = df.to_dict("records")
    done = {(r["modele"], int(r["fold"])) for r in rows
            if "modele" in r and "fold" in r and not pd.isna(r.get("fold"))}
    return rows, done


def save_rows(rows, csv_path=None):
    """Ecrit le CSV avec un ordre de colonnes stable."""
    csv_path = Path(csv_path or cfg.CSV_CV)
    df = pd.DataFrame(rows)
    ordered = [c for c in result_columns() if c in df.columns]
    rest = [c for c in df.columns if c not in ordered]
    df = df[ordered + rest]
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(csv_path, index=False)
    return csv_path


_PAQUETS = {"ultralytics": "ultralytics", "rfdetr": "rfdetr"}


def versions(framework=None):
    """Versions de torch, torchvision, albumentations et du paquet du framework.

    Enregistrees a cote de `repo_commit` : une augmentation ou une metrique peut
    changer d'une version a l'autre sans que le code du benchmark bouge.
    """
    import importlib

    noms = ["torch", "torchvision", "albumentations"]
    if framework in _PAQUETS:
        noms.append(_PAQUETS[framework])
    trouvees = []
    for nom in noms:
        try:
            module = importlib.import_module(nom)
            trouvees.append(f"{nom}={getattr(module, '__version__', '?')}")
        except Exception:
            continue
    return " ".join(trouvees)


def base_row(modele, framework, fold, npos, nneg, **extra):
    """Colonnes communes a toutes les lignes de resultat."""
    batch, accum = cfg.batch_for(framework)
    row = {
        "date": datetime.datetime.now().strftime("%Y-%m-%d %H:%M"),
        "modele": modele, "framework": framework, "fold": int(fold),
        "epochs_budget": cfg.MAX_EPOCHS, "imgsz": cfg.IMGSZ,
        "neg_ratio": cfg.NEG_RATIO,
        "batch": batch, "grad_accum": accum, "batch_effectif": batch * accum,
        "n_pucerons_train": npos, "n_fonds_train": nneg,
        # stopped_early = False signale que le plafond MAX_EPOCHS a ete atteint,
        # donc qu'il est trop bas pour ce modele.
        "best_epoch": -1, "epochs_run": -1, "stopped_early": None,
        "repo_commit": "", "versions": versions(framework), "notes": "",
    }
    row.update(extra)
    return row


# --------------------------------------------------------------- boucle de CV
def run_cv(modele, run_fold, folds=None, csv_path=None, force=False):
    """Lance `run_fold(fold) -> dict` sur chaque fold, avec reprise automatique.

    Une ligne deja presente dans le CSV est sautee (sauf `force=True`), et une
    erreur sur un fold n'interrompt pas les suivants.
    """
    folds = cfg.CV_FOLDS if folds is None else folds
    csv_path = Path(csv_path or cfg.CSV_CV)
    rows, done = load_rows(csv_path)

    for fold in folds:
        if (modele, fold) in done and not force:
            print(f"  skip {modele} fold{fold} (deja dans {csv_path.name})")
            continue
        if force:
            rows = [r for r in rows if not (r.get("modele") == modele
                                            and int(r.get("fold", -1)) == fold)]
        print(f"\n=== {modele} | fold {fold} ===", flush=True)
        t0 = time.time()
        try:
            row = run_fold(fold)
            row.setdefault("train_time_s", round(time.time() - t0, 1))
            rows.append(row)
            save_rows(rows, csv_path)
            print(f"  OK {modele} fold{fold} : map50={row.get('map50_macro')} "
                  f"| map50-95={row.get('map5095_macro')} "
                  f"| best_epoch={row.get('best_epoch')}/{row.get('epochs_run')} "
                  f"| arret anticipe={row.get('stopped_early')} "
                  f"| latence_cpu={row.get('latency_cpu_ms')} ms", flush=True)
        except Exception as e:
            print(f"  ERREUR {modele} fold{fold} : {type(e).__name__}: {e}")
            traceback.print_exc()
            continue

    print(f"\nTermine -> {csv_path}")
    return pd.DataFrame(rows)


# ------------------------------------------------------------------- W&B
def wandb_run(modele, fold, config_extra=None):
    """Ouvre un run W&B (ou renvoie None si desactive/indisponible)."""
    if not cfg.USE_WANDB:
        return None
    try:
        import wandb
        return wandb.init(project=cfg.WANDB_PROJECT, name=f"{modele}_fold{fold}",
                          reinit=True, config={"modele": modele, "fold": fold,
                                               "max_epochs": cfg.MAX_EPOCHS,
                                               "early_stop_patience":
                                                   cfg.EARLY_STOP_PATIENCE,
                                               "early_stop_min_delta":
                                                   cfg.EARLY_STOP_MIN_DELTA,
                                               "imgsz": cfg.IMGSZ,
                                               "neg_ratio": cfg.NEG_RATIO,
                                               **(config_extra or {})})
    except Exception as e:
        print("  W&B indisponible :", e)
        return None


def wandb_finish(run, summary=None, suivi=None):
    """Cloture le run W&B. `suivi` remonte best_epoch, epochs_run, stopped_early."""
    if run is None:
        return
    try:
        if summary:
            run.summary.update({k: v for k, v in summary.items()
                                if isinstance(v, (int, float))})
        if suivi:
            run.summary.update({k: suivi.get(k) for k in
                                ("best_epoch", "epochs_run", "stopped_early")
                                if k in suivi})
        run.finish()
    except Exception:
        pass
