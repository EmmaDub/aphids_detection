# -*- coding: utf-8 -*-
"""Genere les notebooks Colab du benchmark (un par framework).

    python tools/make_notebooks.py

Les notebooks sont volontairement minces : toute la logique vit dans le package
`aphids_det`, ce qui evite de dupliquer le protocole dans 6 fichiers .ipynb.
"""

import json
from pathlib import Path

REPO_URL = "https://github.com/EmmaDub/aphids_detection.git"
REPO_DIR = "/content/aphids_detection"
NB_DIR = Path(__file__).resolve().parent.parent / "notebooks"


def md(text):
    return {"cell_type": "markdown", "metadata": {}, "source": text.strip().splitlines(True)}


def code(text):
    return {"cell_type": "code", "metadata": {}, "execution_count": None,
            "outputs": [], "source": text.strip().splitlines(True)}


# --------------------------------------------------------------- cellules communes
def cell_code_repo():
    return code(f"""
# --- Code du benchmark (package aphids_det) ---
import os, sys
REPO_DIR = "{REPO_DIR}"
if not os.path.exists(REPO_DIR):
    !git clone -q {REPO_URL} {{REPO_DIR}}
else:
    !git -C {{REPO_DIR}} pull -q
sys.path.insert(0, REPO_DIR)
import aphids_det
print("aphids_det", aphids_det.__version__)
""")


CELL_DRIVE = code("""
# --- Connexion Drive ---
from google.colab import drive
drive.mount('/content/drive')
""")

CELL_CONFIG = code("""
# --- Configuration ---
# Les chemins par defaut sont ceux de aphids_det/config.py. Pour les changer,
# decommenter et adapter, puis relancer cfg.refresh().
from pathlib import Path
import aphids_det.config as cfg

# cfg.BASE_DIR  = Path("/content/drive/MyDrive/.../tuile_viz02_640_128")
# cfg.OUT_DIR   = Path("/content/drive/MyDrive/.../puceron_model_article/data")
# cfg.EPOCHS    = 30
# cfg.USE_WANDB = True

cfg.refresh()
cfg.summary()
""")

CELL_FOLDS = code("""
# --- Construction des folds (symlinks locaux, a refaire a chaque session Colab) ---
from aphids_det import folds
folds.build_folds()
""")

CELL_WANDB = code("""
# --- Suivi W&B (facultatif : mettre cfg.USE_WANDB = False pour s'en passer) ---
if cfg.USE_WANDB:
    import wandb
    wandb.login()
    os.environ["WANDB_PROJECT"] = cfg.WANDB_PROJECT
    print("W&B -> projet", cfg.WANDB_PROJECT)
""")

CELL_TAIL = code("""
# --- Etat du CSV de benchmark ---
import pandas as pd
d = pd.read_csv(cfg.CSV_CV)
print(d.groupby("modele")["fold"].count().to_string(), "\\n")
d[["modele", "fold", "map50_macro", "map5095_macro", "latency_cpu_ms",
   "n_params_M", "train_time_s"]].tail(10)
""")


def preamble(title, intro, install_cell, folds=True, wandb=True):
    cells = [md(f"# {title}\n\n{intro}"), install_cell, CELL_DRIVE, cell_code_repo(),
             CELL_CONFIG]
    if folds:
        cells.append(CELL_FOLDS)
    if wandb:
        cells.append(CELL_WANDB)
    return cells


def notebook(cells):
    return {
        "cells": cells,
        "metadata": {
            "accelerator": "GPU",
            "colab": {"provenance": [], "gpuType": "T4"},
            "kernelspec": {"display_name": "Python 3", "name": "python3"},
            "language_info": {"name": "python"},
        },
        "nbformat": 4, "nbformat_minor": 0,
    }


# ------------------------------------------------------------------ notebooks
NOTEBOOKS = {}

NOTEBOOKS["00_preparation_folds.ipynb"] = notebook(
    preamble(
        "00 - Preparation : folds, verification, table d'augmentation",
        "A lancer une fois avant les notebooks de modeles. Construit les folds "
        "depuis le CSV de split, fige les listes de train partagees par tous les "
        "modeles, verifie la plomberie et exporte la table d'augmentation.\n\n"
        "**Le fold 5 est le test : il n'est jamais utilise par la comparaison.**",
        code("""
# --- Installation (package partage uniquement) ---
!pip install -q pycocotools openpyxl wandb
"""),
        wandb=False,
    ) + [
        code("""
# --- Verification de la plomberie (sans GPU ni donnees) ---
!python {REPO_DIR}/tests/test_pipeline.py
"""),
        md("## Datasets COCO et table d'augmentation"),
        code("""
# --- Construction des datasets COCO des 5 folds (les deux arborescences) ---
# Optionnel ici : chaque notebook de modele le fait pour ses propres folds.
from aphids_det import cocoify
for fold in cfg.CV_FOLDS:
    for layout in ("flat", "coco"):
        ds, npos, nneg = cocoify.build_coco_fold(fold, layout=layout, verbose=False)
    print(f"fold {fold} : train {npos} pucerons / {nneg} fonds -> {ds.parent}")
"""),
        code("""
# --- Table de correspondance des augmentations (CSV a cote des resultats) ---
from aphids_det import augment
augment.write_csv()
augment.table()
"""),
    ])

NOTEBOOKS["01_ultralytics.ipynb"] = notebook(
    preamble(
        "01 - YOLO26n / YOLO11n / YOLO12n (Ultralytics)",
        "Trois modeles x 5 folds. Augmentation = `augment.AUG` appliquee telle "
        "quelle ; tous les autres hyperparametres restent les defauts Ultralytics. "
        "Les metriques viennent de l'evaluateur COCO unifie, pas de `model.val()`.",
        code("""
# --- Installation ---
!pip install -q ultralytics wandb pycocotools
"""),
    ) + [
        md("## Comparaison en validation croisee\n\n"
           "Reprise automatique : un (modele, fold) deja present dans le CSV est saute."),
        code("""
from aphids_det.runners import ultralytics_runner
df = ultralytics_runner.run_cv()      # folds 0-4 pour les 3 modeles
"""),
        CELL_TAIL,
    ])

NOTEBOOKS["02_rfdetr.ipynb"] = notebook(
    preamble(
        "02 - RF-DETR Nano (package rfdetr de Roboflow)",
        "Dataset COCO 'plat' (train/valid/test) construit depuis les memes listes "
        "de tuiles. Batch 8 x 4 = 32 effectif. Augmentation : `augment.AUG_RFDETR`.",
        code("""
# --- Installation ---
!pip install -q "rfdetr[train,loggers]" pycocotools wandb
"""),
    ) + [
        code("""
from aphids_det.runners import rfdetr_runner
df = rfdetr_runner.run_cv()
"""),
        CELL_TAIL,
    ])

NOTEBOOKS["03_rtdetr_r18.ipynb"] = notebook(
    preamble(
        "03 - RT-DETR-R18 (depot officiel lyuwenyu/RT-DETR)",
        "Le depot est clone, ses dependances installees, et deux transforms "
        "ajoutees a son registre (flip vertical, ColorJitter).\n\n"
        "**Variante par defaut : `rtdetr_v2`** (`rtdetrv2_pytorch`, backbone R18). "
        "La version v1 (`rtdetr_pytorch`, le RT-DETR-R18 de l'article) importe "
        "`torchvision.datapoints`, supprime depuis torchvision 0.17 : elle ne "
        "demarre pas sur un Colab recent. Pour l'essayer malgre tout : "
        "`rtdetr_runner.run_cv(variant='rtdetr_v1')` dans un environnement avec "
        "torchvision <= 0.16.",
        code("""
# --- Installation (le depot installe ses propres dependances au 1er appel) ---
!pip install -q pycocotools wandb
"""),
    ) + [
        code("""
from aphids_det.runners import rtdetr_runner
df = rtdetr_runner.run_cv()           # variante rtdetr_v2 (R18)
"""),
        CELL_TAIL,
    ])

NOTEBOOKS["04_dfine_n.ipynb"] = notebook(
    preamble(
        "04 - D-FINE-N (depot officiel Peterande/D-FINE)",
        "Config `dfine_hgnetv2_n_coco`, poids COCO `dfine_n_coco.pth` en "
        "fine-tuning. Memes transforms ajoutees que pour RT-DETR.",
        code("""
# --- Installation (le depot installe ses propres dependances au 1er appel) ---
!pip install -q pycocotools wandb
"""),
    ) + [
        code("""
from aphids_det.runners import dfine_runner
df = dfine_runner.run_cv()
"""),
        CELL_TAIL,
    ])

NOTEBOOKS["05_yolox_nano.ipynb"] = notebook(
    preamble(
        "05 - YOLOX-Nano (depot officiel Megvii-BaseDetection/YOLOX)",
        "YOLOX n'est plus maintenu : il demande `numpy < 2`. La cellule "
        "d'installation le retrograde, **ce qui impose de redemarrer le runtime** "
        "(Execution > Redemarrer la session) avant de continuer.\n\n"
        "Resolution portee a 640 (au lieu de 416 par defaut) pour rester "
        "comparable ; mixup desactive ; flip vertical et gains HSV ajoutes.",
        code("""
# --- Installation : redemarrer le runtime apres cette cellule ---
!pip install -q "numpy<2" pycocotools wandb
"""),
    ) + [
        code("""
from aphids_det.runners import yolox_runner
df = yolox_runner.run_cv()
"""),
        CELL_TAIL,
    ])

NOTEBOOKS["06_synthese.ipynb"] = notebook(
    [md("# 06 - Synthese : moyennes par modele et classeur Excel\n\n"
        "A lancer une fois les 7 modeles passes. Produit un classeur a 4 feuilles : "
        "hyperparametres, augmentations, resultats par fold, moyennes par modele."),
     code("""
# --- Installation ---
!pip install -q openpyxl pandas
"""),
     CELL_DRIVE, cell_code_repo(), CELL_CONFIG,
     code("""
# --- Moyennes par modele ---
from aphids_det import report
moy = report.summary()
moy[["modele", "n_folds", "map50_macro", "map5095_macro", "latency_cpu_ms",
     "n_params_M", "size_MB", "train_time_s"]]
"""),
     code("""
# --- Classeur Excel complet ---
report.to_excel()
"""),
     code("""
# --- Detail par classe ---
import pandas as pd
d = pd.read_csv(cfg.CSV_CV)
cls = [c for c in d.columns if c.startswith(tuple(cfg.CLASS_NAMES.values()))]
d.groupby("modele")[cls].mean().round(3)
"""),
     ])


def main():
    NB_DIR.mkdir(parents=True, exist_ok=True)
    for name, nb in NOTEBOOKS.items():
        (NB_DIR / name).write_text(json.dumps(nb, indent=1, ensure_ascii=False),
                                   encoding="utf-8")
        print("ecrit :", NB_DIR / name)


if __name__ == "__main__":
    main()
