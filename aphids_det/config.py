# -*- coding: utf-8 -*-
"""Parametres globaux du benchmark de detection de pucerons.

Tout est modifiable depuis un notebook, puis `refresh()` recalcule les chemins
derives et cree les dossiers :

    import aphids_det.config as cfg
    cfg.OUT_DIR = Path("/content/drive/MyDrive/.../puceron_model_article/data")
    cfg.refresh()

Conventions du protocole (identiques aux notebooks d'origine) :
  - 6 folds dans le CSV de split ; folds 0-4 = validation croisee, fold 5 = test
    jamais utilise pendant la comparaison ;
  - le train d'un fold = les 4 autres folds + `_extra_train` (renfort "cell"),
    avec les tuiles de fond sous-echantillonnees a NEG_RATIO x positifs ;
  - la liste de train de chaque fold est FIGEE sur le Drive et partagee par tous
    les modeles -> tous les frameworks voient exactement les memes images.
"""

from pathlib import Path

# ============================================================================
# 1. SOURCES DE DONNEES (Drive) -- A ADAPTER
# ============================================================================
BASE_DIR = Path("/content/drive/MyDrive/Emma/puceron_model_2026/"
                "puceron_model_E2026_3/data/tuile_viz02_640_128")
SPLIT_DIR = BASE_DIR / "split"
CELL_DIR = Path("/content/drive/MyDrive/Emma/puceron_model_2026/"
                "puceron_model_E2026_2/data/cell/tuile_viz02_640_128_cell")

MAIN_IMAGES_DIR = BASE_DIR / "images"
BG_DIR = Path("/content/drive/MyDrive/Emma/puceron_model_2026/"
              "puceron_model_E2026_2/data/images_complete")
# Archive de secours si BG_DIR est vide (extraite une seule fois en local).
BG_ZIP = BG_DIR.parent / "tuile_viz02_640_128_background.zip"


def V(csv, labels, extra_train=None):
    """Decrit une variante de labels (identique aux notebooks)."""
    labels = Path(labels)
    if not labels.is_absolute():
        labels = BASE_DIR / labels
    variant = {"csv": Path(csv), "labels_dir": labels}
    if extra_train is not None:
        variant["extra_train"] = {
            "csv": Path(extra_train["csv"]),
            "labels_dir": Path(extra_train["labels_dir"]),
            "images_dir": Path(extra_train["images_dir"]),
        }
    return variant


EXPERIMENT_NAME = "yolo_neg1"
VARIANTS = {
    "labels_cell_20": V(
        SPLIT_DIR / "split_assignments_all_background.csv",
        "labels_visible_20",
        extra_train={
            "csv": CELL_DIR / "split" / "split_assignments.csv",
            "labels_dir": CELL_DIR / "labels_cell_20",
            "images_dir": CELL_DIR / "images_lookmatched3",
        },
    ),
}
SEARCH_VARIANT = "labels_cell_20"

# ============================================================================
# 2. PROTOCOLE
# ============================================================================
CLASS_NAMES = {0: "Apterous_aphid", 1: "Alate_aphid"}
N_CV_FOLDS = 5              # folds 0..4 utilises en CV
TEST_FOLD = 5               # fold reserve au test final (jamais en CV)
CV_FOLDS = list(range(N_CV_FOLDS))
NEG_RATIO = 3               # fonds gardes dans le train = 3 x tuiles positives
SEED = 42
TILE_SIZE = 640             # toutes les tuiles font 640x640 (evite d'ouvrir chaque image)

# ============================================================================
# 3. BUDGET D'ENTRAINEMENT (commun a tous les modeles)
# ============================================================================
EPOCHS = 30
IMGSZ = 640
PATIENCE = 5                # early stopping natif (Ultralytics, RF-DETR) ; cf. README
EFFECTIVE_BATCH = 32        # batch effectif vise pour tous les modeles

# Batch physique par framework. RF-DETR compense par accumulation de gradient ;
# RT-DETR, D-FINE et YOLOX n'ont pas d'accumulation -> batch physique = effectif.
BATCH_PHYSICAL = {
    "ultralytics": 32,
    "rfdetr": 8,            # 8 x 4 = 32 (valeur validee sur T4)
    "rtdetr": 16,           # R18 @640 : 32 passe rarement sur une T4
    "dfine": 32,
    "yolox": 32,
}
GRAD_ACCUM = {"rfdetr": 4}  # frameworks supportant l'accumulation
AUTO_BATCH_RETRY = 2        # nb de divisions par 2 du batch en cas d'OOM (0 = desactive)

# ============================================================================
# 3 bis. HARMONISATION DE L'AUGMENTATION
# ============================================================================
# Geometrie (echelle et translation) de RT-DETR et D-FINE :
#   "reference" : les transforms natives des depots sont conservees mais leurs
#                 bornes sont calees sur AUG -> echelle x[0.75, 1.25] comme les
#                 YOLO, au lieu de x[0.25, 3.3] par defaut ;
#   "affine"    : ZoomOut et IoUCrop remplaces par un RandomAffine portant
#                 exactement les parametres d'Ultralytics ;
#   "natif"     : amplitudes d'origine des depots (non comparables).
DETR_GEOM = "reference"

# Part de son aire d'origine qu'une boite doit conserver apres augmentation pour
# rester annotee. Ultralytics applique 0.10 en dur : le benchmark le porte a
# cette valeur. Cf. docs/AUGMENTATION.md pour ce que chaque depot fait.
MIN_VISIBILITY = 0.20

# ============================================================================
# 4. INFERENCE ET METRIQUES (identiques pour tous -> chiffres comparables)
# ============================================================================
CONF_EVAL = 0.001           # seuil de confiance pour l'export des predictions (mAP)
NMS_IOU = 0.7               # IoU de NMS pour les modeles a NMS (YOLO, YOLOX)
MAX_DET = 300               # detections max par tuile
CONF_PR = 0.25              # seuil de confiance pour P / R / F1 / TP / FP / FN
IOU_PR = 0.5                # IoU d'appariement pour P / R / F1
LATENCY_WARMUP = 10
LATENCY_ITERS = 50

# ============================================================================
# 5. MODELES COMPARES
# ============================================================================
# nom -> (framework, specification propre au framework)
MODELS = {
    "YOLO26n":     ("ultralytics", "yolo26n.pt"),
    "YOLO11n":     ("ultralytics", "yolo11n.pt"),
    "YOLO12n":     ("ultralytics", "yolo12n.pt"),
    "RF-DETR-N":   ("rfdetr", "RFDETRNano"),
    "RT-DETR-R18": ("rtdetr", "rtdetr_r18vd_6x_coco"),
    "D-FINE-N":    ("dfine", "dfine_hgnetv2_n_coco"),
    "YOLOX-Nano":  ("yolox", "yolox_nano"),
}

# ============================================================================
# 6. SORTIES
# ============================================================================
# Racine Drive des resultats (persistante entre les sessions Colab).
OUT_DIR = Path("/content/drive/MyDrive/Emma/puceron_model_2026/"
               "puceron_model_article/data")
# Racine locale de travail (rapide, ephemere).
WORK_ROOT = Path("/content/aphids_work")

USE_WANDB = True
WANDB_PROJECT = "comparaison_pucerons_detection"

# --- Chemins derives (recalcules par refresh()) -----------------------------
FOLDS_ROOT = YAML_DIR = COCO_ROOT = REPOS_DIR = STAGE_LOCAL = None
CSV_CV = SAVE_DIR = SPLITS_FROZEN = PRED_DIR = HP_CSV = AUG_CSV = None


def refresh(mkdirs=True):
    """Recalcule les chemins derives de OUT_DIR / WORK_ROOT et cree les dossiers."""
    global FOLDS_ROOT, YAML_DIR, COCO_ROOT, REPOS_DIR, STAGE_LOCAL
    global CSV_CV, SAVE_DIR, SPLITS_FROZEN, PRED_DIR, HP_CSV, AUG_CSV

    FOLDS_ROOT = WORK_ROOT / "folds" / EXPERIMENT_NAME / SEARCH_VARIANT
    YAML_DIR = WORK_ROOT / "yaml" / EXPERIMENT_NAME
    COCO_ROOT = WORK_ROOT / "coco"
    REPOS_DIR = WORK_ROOT / "repos"
    STAGE_LOCAL = WORK_ROOT / "stage"

    # CSV unique du benchmark (evaluation COCO unifiee). A ne pas confondre avec
    # comparaison_modeles_cv.csv des anciens notebooks (metriques natives).
    CSV_CV = OUT_DIR / "benchmark_detection_cv.csv"
    SAVE_DIR = OUT_DIR / "modeles_detection"
    SPLITS_FROZEN = OUT_DIR / "splits_figes"      # partage avec les anciens notebooks
    PRED_DIR = OUT_DIR / "predictions_coco"
    HP_CSV = OUT_DIR / "hyperparametres_detection.csv"
    AUG_CSV = OUT_DIR / "augmentations_detection.csv"

    if mkdirs:
        for d in (FOLDS_ROOT, YAML_DIR, COCO_ROOT, REPOS_DIR, STAGE_LOCAL,
                  OUT_DIR, SAVE_DIR, SPLITS_FROZEN, PRED_DIR):
            Path(d).mkdir(parents=True, exist_ok=True)


def batch_for(framework):
    """(batch physique, accumulation de gradient) pour un framework."""
    b = BATCH_PHYSICAL.get(framework, EFFECTIVE_BATCH)
    return b, GRAD_ACCUM.get(framework, 1)


def summary():
    """Affiche la configuration effective (a lancer en debut de notebook)."""
    print(f"Variante        : {SEARCH_VARIANT}")
    print(f"Classes         : {CLASS_NAMES}")
    print(f"CV              : folds {CV_FOLDS} (fold {TEST_FOLD} = test, non utilise)")
    print(f"Budget          : {EPOCHS} epochs | imgsz {IMGSZ} | "
          f"batch effectif {EFFECTIVE_BATCH} | neg_ratio {NEG_RATIO}")
    print(f"Evaluation      : COCO unifiee (conf {CONF_EVAL}, NMS IoU {NMS_IOU}, "
          f"maxDets {MAX_DET}) | P/R/F1 a conf {CONF_PR}, IoU {IOU_PR}")
    print(f"Travail local   : {WORK_ROOT}")
    print(f"Resultats Drive : {OUT_DIR}")
    print(f"CSV benchmark   : {CSV_CV}")


refresh(mkdirs=False)
