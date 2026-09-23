# -*- coding: utf-8 -*-
"""Depots externes : clone, poids COCO, lancement de commandes.

RT-DETR, D-FINE et YOLOX ne sont pas des packages pip : ils s'utilisent en
clonant le depot officiel et en lancant leur script d'entrainement. Ce module
centralise ces operations et journalise le commit exact utilise (colonne
`repo_commit` du CSV de resultats) pour que la comparaison reste reproductible.
"""

import os
import re
import subprocess
import sys
import urllib.request
from pathlib import Path

from . import config as cfg

# --- Depots officiels -------------------------------------------------------
REPOS = {
    "rtdetr": {"url": "https://github.com/lyuwenyu/RT-DETR.git", "dir": "RT-DETR"},
    "dfine": {"url": "https://github.com/Peterande/D-FINE.git", "dir": "D-FINE"},
    "yolox": {"url": "https://github.com/Megvii-BaseDetection/YOLOX.git", "dir": "YOLOX"},
}

# --- Poids pre-entraines COCO (verifies le 2026-09-20) ----------------------
# Si un lien casse, le recuperer dans le README du depot correspondant.
WEIGHTS = {
    "rtdetr_v2_r18": ("https://github.com/lyuwenyu/storage/releases/download/v0.1/"
                      "rtdetrv2_r18vd_120e_coco.pth"),
    "rtdetr_v1_r18": ("https://github.com/lyuwenyu/storage/releases/download/v0.1/"
                      "rtdetr_r18vd_dec3_6x_coco_from_paddle.pth"),
    "dfine_n": ("https://github.com/Peterande/storage/releases/download/dfinev1.0/"
                "dfine_n_coco.pth"),
    "yolox_nano": ("https://github.com/Megvii-BaseDetection/YOLOX/releases/download/"
                   "0.1.1rc0/yolox_nano.pth"),
}

OOM_PATTERNS = re.compile(r"out of memory|CUDA out of memory|CUBLAS_STATUS_ALLOC_FAILED",
                          re.IGNORECASE)


# ------------------------------------------------------------------ commandes
def run(cmd, cwd=None, env=None, check=True, echo=True, surveillance=None):
    """Lance une commande en streamant sa sortie. Renvoie (code, sortie).

    `surveillance` (cf. runners/arret_anticipe.SurveillanceJournal) peut mettre
    fin au processus en cours de route : un arret anticipe n'est alors PAS une
    erreur, meme si le code de retour est non nul.
    """
    if isinstance(cmd, str):
        cmd = cmd.split()
    cmd = [str(c) for c in cmd]
    if echo:
        print("$", " ".join(cmd), flush=True)
    full_env = {**os.environ, **(env or {})}
    proc = subprocess.Popen(cmd, cwd=str(cwd) if cwd else None, env=full_env,
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            text=True, bufsize=1)
    if surveillance is not None:
        surveillance.demarrer(proc)
    lines = []
    try:
        for line in proc.stdout:
            lines.append(line)
            if echo:
                sys.stdout.write(line)
                sys.stdout.flush()
        proc.wait()
    finally:
        if surveillance is not None:
            surveillance.arreter()
    out = "".join(lines)
    if surveillance is not None and surveillance.interrompu:
        return 0, out                      # arret voulu : on ne leve pas d'erreur
    if check and proc.returncode != 0:
        raise RuntimeError(f"Commande echouee (code {proc.returncode}) : {' '.join(cmd)}\n"
                           f"{out[-3000:]}")
    return proc.returncode, out


def run_with_oom_retry(build_cmd, batch, cwd=None, retries=None, min_batch=1,
                       surveillance=None):
    """Lance `build_cmd(batch)` et divise le batch par 2 en cas d'OOM CUDA.

    `surveillance` est transmise a `run` : elle peut arreter l'entrainement des
    que l'early stopping se declenche.

    Renvoie (sortie, batch reellement utilise).
    """
    retries = cfg.AUTO_BATCH_RETRY if retries is None else retries
    for attempt in range(retries + 1):
        code, out = run(build_cmd(batch), cwd=cwd, check=False,
                        surveillance=surveillance)
        if code == 0:
            return out, batch
        if OOM_PATTERNS.search(out) and attempt < retries and batch // 2 >= min_batch:
            batch = batch // 2
            print(f"\n  !! OOM CUDA -> nouvelle tentative avec batch={batch}\n", flush=True)
            continue
        raise RuntimeError(f"Entrainement echoue (code {code}). Fin de sortie :\n{out[-3000:]}")
    raise RuntimeError("Entrainement echoue apres toutes les tentatives.")


# ----------------------------------------------------------------------- git
def clone(framework, branch=None):
    """Clone (une fois) le depot officiel et renvoie son chemin local."""
    spec = REPOS[framework]
    dst = Path(cfg.REPOS_DIR) / spec["dir"]
    if not (dst / ".git").exists():
        dst.parent.mkdir(parents=True, exist_ok=True)
        cmd = ["git", "clone", "--depth", "1"]
        if branch:
            cmd += ["--branch", branch]
        cmd += [spec["url"], str(dst)]
        run(cmd)
    else:
        print(f"Depot deja clone : {dst}")
    return dst


def git_commit(repo_dir):
    """Hash court du commit clone (pour la tracabilite des resultats)."""
    try:
        code, out = run(["git", "rev-parse", "--short", "HEAD"], cwd=repo_dir,
                        check=False, echo=False)
        return out.strip() if code == 0 else ""
    except Exception:
        return ""


# --------------------------------------------------------------------- poids
def download(url, dst):
    """Telecharge un fichier s'il est absent. Message explicite si le lien casse."""
    dst = Path(dst)
    if dst.exists() and dst.stat().st_size > 0:
        print(f"Poids deja presents : {dst}")
        return dst
    dst.parent.mkdir(parents=True, exist_ok=True)
    print(f"Telechargement {url}")
    try:
        urllib.request.urlretrieve(url, dst)
    except Exception as e:
        raise RuntimeError(
            f"Telechargement impossible ({e}).\nURL : {url}\n"
            "Les depots deplacent parfois leurs poids : recuperer le lien a jour dans "
            "le README du depot officiel et mettre a jour aphids_det/external.WEIGHTS."
        ) from e
    print(f"  -> {dst} ({dst.stat().st_size / 1e6:.1f} Mo)")
    return dst


def pretrained(key):
    """Telecharge (une fois) des poids COCO de reference et renvoie le chemin."""
    url = WEIGHTS[key]
    return download(url, Path(cfg.REPOS_DIR) / "weights" / Path(url).name)


# ------------------------------------------------------------------- imports
def add_to_path(repo_dir):
    """Rend un depot clone importable (`import src...`) depuis le notebook."""
    repo_dir = str(repo_dir)
    if repo_dir not in sys.path:
        sys.path.insert(0, repo_dir)
    return repo_dir
