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
# Depot PRIVE ? Deposer un jeton GitHub dans les secrets Colab (icone cle a
# gauche) sous le nom GITHUB_TOKEN, avec l'acces "Contents: read" sur ce depot.
# Le jeton n'est jamais ecrit dans le notebook ni affiche.
# Depot PUBLIC ? Rien a faire, la cellule fonctionne telle quelle.
import os, subprocess, sys

REPO_DIR = "{REPO_DIR}"
REPO_URL = "{REPO_URL}"
os.environ["GIT_TERMINAL_PROMPT"] = "0"   # echouer plutot que demander un mot de passe

url = REPO_URL
try:
    from google.colab import userdata
    jeton = userdata.get("GITHUB_TOKEN")
    if jeton:
        url = REPO_URL.replace("https://", f"https://{{jeton}}@")
        print("jeton GITHUB_TOKEN trouve dans les secrets Colab")
except Exception:
    pass

if os.path.exists(os.path.join(REPO_DIR, ".git")):
    subprocess.run(["git", "-C", REPO_DIR, "remote", "set-url", "origin", url],
                   capture_output=True, text=True)
    r = subprocess.run(["git", "-C", REPO_DIR, "pull", "-q"],
                       capture_output=True, text=True)
else:
    r = subprocess.run(["git", "clone", "-q", url, REPO_DIR],
                       capture_output=True, text=True)

if r.returncode != 0:
    erreur = (r.stderr or r.stdout or "").replace(url, REPO_URL).strip()
    raise RuntimeError(
        "Recuperation du code impossible :\\n" + erreur + "\\n\\n"
        "Si le message parle d'identifiant ('could not read Username'), le depot "
        "est prive et Colab n'a pas d'acces. Au choix :\\n"
        "  1. rendre le depot public (Settings > General > Change visibility) ;\\n"
        "  2. creer un jeton sur github.com/settings/tokens (fine-grained, acces "
        "'Contents: read' sur ce depot) et le deposer dans les secrets Colab "
        "sous le nom GITHUB_TOKEN, avec l'acces notebook active ;\\n"
        "  3. copier le dossier du depot sur le Drive et remplacer REPO_DIR par "
        "son chemin.")

sys.path.insert(0, REPO_DIR)
import aphids_det
print("aphids_det", aphids_det.__version__, "|", REPO_DIR)
""")


CELL_DRIVE = code("""
# --- Connexion Drive ---
from google.colab import drive
drive.mount('/content/drive')
""")

CELL_CONFIG = code("""
# --- CONFIG DONNEES (source des tuiles + variante labels_cell_20) ---
# Memes chemins que comparaison_modeles_ultralytics.ipynb : c'est ici, et nulle
# part ailleurs, qu'on change de jeu de donnees.
from pathlib import Path
import aphids_det.config as cfg

cfg.BASE_DIR = Path("/content/drive/MyDrive/Emma/puceron_model_2026/"
                    "puceron_model_E2026_3/data/tuile_viz02_640_128")
cfg.SPLIT_DIR = cfg.BASE_DIR / "split"
cfg.CELL_DIR = Path("/content/drive/MyDrive/Emma/puceron_model_2026/"
                    "puceron_model_E2026_2/data/cell/tuile_viz02_640_128_cell")
cfg.MAIN_IMAGES_DIR = cfg.BASE_DIR / "images"
cfg.BG_DIR = Path("/content/drive/MyDrive/Emma/puceron_model_2026/"
                  "puceron_model_E2026_2/data/images_complete")
cfg.BG_ZIP = cfg.BG_DIR.parent / "tuile_viz02_640_128_background.zip"

cfg.EXPERIMENT_NAME = "yolo_neg1"
cfg.SEARCH_VARIANT = "labels_cell_20"
cfg.VARIANTS = {
    "labels_cell_20": cfg.V(
        cfg.SPLIT_DIR / "split_assignments_all_background.csv", "labels_visible_20",
        extra_train={"csv":        cfg.CELL_DIR / "split" / "split_assignments.csv",
                     "labels_dir": cfg.CELL_DIR / "labels_cell_20",
                     "images_dir": cfg.CELL_DIR / "images_lookmatched3"}),
}

cfg.CLASS_NAMES = {0: "Apterous_aphid", 1: "Alate_aphid"}
cfg.N_CV_FOLDS = 5          # folds 0-4 en validation croisee
cfg.TEST_FOLD = 5           # fold 5 : test, jamais utilise ici
cfg.CV_FOLDS = list(range(cfg.N_CV_FOLDS))
cfg.NEG_RATIO = 3

# --- SORTIES (Drive) et budget ---
cfg.OUT_DIR = Path("/content/drive/MyDrive/Emma/puceron_model_2026/"
                   "puceron_model_article/data")
cfg.EPOCHS = 30
cfg.IMGSZ = 640
cfg.PATIENCE = 5
cfg.SEED = 42
cfg.USE_WANDB = True
cfg.WANDB_PROJECT = "comparaison_pucerons_detection"

cfg.refresh()
cfg.summary()
""")

CELL_CHECK_PATHS = code("""
# --- Verification des chemins Drive avant de lancer quoi que ce soit ---
attendus = {
    "tuiles (images)": cfg.MAIN_IMAGES_DIR,
    "labels": cfg.VARIANTS[cfg.SEARCH_VARIANT]["labels_dir"],
    "CSV de split": cfg.VARIANTS[cfg.SEARCH_VARIANT]["csv"],
    "fonds (images_complete)": cfg.BG_DIR,
    "renfort cell : images": cfg.VARIANTS[cfg.SEARCH_VARIANT]["extra_train"]["images_dir"],
    "renfort cell : labels": cfg.VARIANTS[cfg.SEARCH_VARIANT]["extra_train"]["labels_dir"],
    "renfort cell : CSV": cfg.VARIANTS[cfg.SEARCH_VARIANT]["extra_train"]["csv"],
    "sorties (Drive)": cfg.OUT_DIR,
}
for nom, p in attendus.items():
    p = Path(p)
    etat = "OK     " if p.exists() else "MANQUE "
    extra = ""
    if p.is_dir():
        try:
            extra = f"  ({sum(1 for _ in p.iterdir())} entrees)"
        except OSError:
            pass
    print(f"{etat}{nom:28s} {p}{extra}")
if not Path(cfg.BG_DIR).exists():
    print(f"\\n(fonds absents : l'archive {cfg.BG_ZIP} sera extraite en local)")
""")

CELL_FOLDS = code("""
# --- Construction des folds (symlinks locaux, a refaire a chaque session Colab) ---
# Les listes de train figees sur le Drive contiennent des chemins absolus ecrits
# par la session qui les a creees : ils sont automatiquement regreffes sur la
# racine de folds courante, la selection de tuiles reste donc identique.
from aphids_det import folds
folds.build_folds()
""")

CELL_DIAGNOSE = code("""
# --- Diagnostic : a lancer si un fold parait vide ---
# Affiche images/labels par fold, et ce que donnent les listes figees du Drive
# une fois regreffees sur la racine courante.
folds.diagnose(fold=0)
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
             CELL_CONFIG, CELL_CHECK_PATHS]
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
        CELL_DIAGNOSE,
        code("""
# --- Verification de la plomberie (sans GPU ni donnees) ---
!python {REPO_DIR}/tests/test_pipeline.py
!python {REPO_DIR}/tests/test_rebase.py
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

NOTEBOOKS["07_visualisation_augmentations.ipynb"] = notebook(
    preamble(
        "07 - Voir les augmentations des quatre pipelines cote a cote",
        "Une ligne par pipeline, une colonne par tirage aleatoire, sur la meme "
        "tuile. Chaque ligne execute le **vrai** code d'augmentation du "
        "framework : `YOLODataset` d'Ultralytics (mosaique comprise), les "
        "transforms albumentations de `aug_config` pour RF-DETR, les classes "
        "`torchvision.transforms.v2` de la liste d'ops des depots pour "
        "RT-DETR/D-FINE, et `random_affine` + `augment_hsv` de YOLOX orchestres "
        "comme sa `MosaicDetection`.\n\n"
        "Les quatre cohabitent dans ce runtime parce qu'aucun entrainement n'a "
        "lieu : YOLOX est clone **sans etre installe**, seules ses fonctions "
        "numpy/cv2 sont importees.",
        code("""
# --- Installation (aucun entrainement ici : les 4 pipelines coexistent) ---
!pip install -q ultralytics albumentations loguru thop tabulate psutil pycocotools
"""),
        wandb=False,
    ) + [
        CELL_DIAGNOSE,
        md("## Figure comparative\n\n"
           "A regarder en priorite : la **mosaique** (presente pour YOLO et YOLOX, "
           "absente des trois DETR), les **miroirs verticaux** (ajoutes par le "
           "benchmark a RT-DETR, D-FINE et YOLOX), l'amplitude de la variation "
           "**HSV**, et ce que `RandomZoomOut` + `RandomIoUCrop` produisent chez "
           "les DETR a la place de `translate` et `scale`."),
        code("""
from aphids_det import visualize
fig = visualize.compare(fold=0, n_aug=4, seed=0, save=True)
"""),
        md("## Plusieurs tuiles\n\n"
           "Meme figure sur d'autres tuiles : une seule tuile ne suffit pas a "
           "juger d'une augmentation aleatoire."),
        code("""
tuiles, vivier = visualize.sample_tiles(fold=0, n=3, seed=1, min_boxes=2)
for t in tuiles:
    visualize.compare(fold=0, tile=t, pool=vivier, n_aug=4, seed=1, save=True)
"""),
        md("## Un seul pipeline, plus de tirages\n\n"
           "Pour inspecter un framework en particulier."),
        code("""
visualize.compare(fold=0, n_aug=6, seed=2,
                  pipelines={"YOLOX-Nano": visualize.PIPELINES["YOLOX-Nano"]})
"""),
        md("## Comparer les trois geometries possibles pour RT-DETR / D-FINE\n\n"
           "`cfg.DETR_GEOM` decide de l'amplitude d'echelle et de translation des "
           "deux DETR :\n\n"
           "- **`reference`** (defaut) : les transforms natives des depots, bornes "
           "calees sur `scale = 0.25` -> objet dans x[0.75, 1.25], comme les YOLO ;\n"
           "- **`affine`** : `RandomZoomOut` et `RandomIoUCrop` remplaces par un "
           "`RandomAffine` portant exactement `translate = 0.1` et "
           "`scale = (0.75, 1.25)` ;\n"
           "- **`natif`** : les amplitudes d'origine des depots, x[0.25, 3.3].\n\n"
           "La figure ci-dessous met les trois cote a cote sur la meme tuile. "
           "Choisir, puis fixer `cfg.DETR_GEOM` dans la cellule de configuration "
           "des notebooks d'entrainement."),
        code("""
def geometrie(mode):
    def pipeline(tile, n_aug, pool, seed=0):
        cfg.DETR_GEOM = mode          # lu a chaque appel par patch_ops
        return visualize.aug_detr(tile, n_aug, pool, seed=seed)
    return pipeline

visualize.compare(fold=0, n_aug=4, seed=0, pipelines={
    "RT-DETR / D-FINE (reference)": geometrie("reference"),
    "RT-DETR / D-FINE (affine)":    geometrie("affine"),
    "RT-DETR / D-FINE (natif)":     geometrie("natif"),
})
cfg.DETR_GEOM = "reference"           # on remet le defaut
"""),
        code("""
# --- Table des correspondances, a mettre en regard de la figure ---
from aphids_det import augment
augment.table()
"""),
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
