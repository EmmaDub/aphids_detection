# -*- coding: utf-8 -*-
"""Augmentation de donnees : reference unique + traductions par framework.

Regle du benchmark : les hyperparametres d'entrainement restent ceux par defaut
de chaque modele ; SEULE l'augmentation est harmonisee. La reference est le dict
`AUG` d'Ultralytics ; chaque framework recoit la traduction la plus proche que
son pipeline permet, et les ecarts residuels sont documentes dans `table()`
(exporte en CSV par `write_csv()` et repris dans docs/AUGMENTATION.md).
"""

import pandas as pd

from . import config as cfg

# ============================================================================
# Reference : Ultralytics (applique tel quel a YOLO26n / YOLO11n / YOLO12n)
# ============================================================================
AUG = {
    "hsv_h": 0, "hsv_s": 0.2, "hsv_v": 0.2,
    "degrees": 0, "translate": 0.1, "scale": 0.25,
    "shear": 0, "perspective": 0,
    "flipud": 0.5, "fliplr": 0.5,
    "mosaic": 1, "mixup": 0.0, "cutmix": 0,
    "bgr": 0.0, "auto_augment": None,
}

# ============================================================================
# RF-DETR : transforms albumentations passees a `aug_config`
# ============================================================================
AUG_RFDETR = {
    "HorizontalFlip": {"p": AUG["fliplr"]},
    "VerticalFlip": {"p": AUG["flipud"]},
    "RandomBrightnessContrast": {"brightness_limit": AUG["hsv_v"],
                                 "contrast_limit": 0.0, "p": 1.0},
    "ColorJitter": {"brightness": 0.0, "contrast": 0.0,
                    "saturation": AUG["hsv_s"], "hue": AUG["hsv_h"], "p": 1.0},
}

# ============================================================================
# RT-DETR / D-FINE : ops inserees dans la liste de transforms du depot
# (le reste de la liste d'origine est conserve tel quel, cf. runners/detr_repo.py)
# ============================================================================
DETR_COLOR_OP = {"type": "ColorJitter",
                 "brightness": AUG["hsv_v"], "contrast": 0.0,
                 "saturation": AUG["hsv_s"], "hue": AUG["hsv_h"]}
DETR_HFLIP_OP = {"type": "RandomHorizontalFlip", "p": AUG["fliplr"]}
DETR_VFLIP_OP = {"type": "RandomVerticalFlip", "p": AUG["flipud"]}

# ============================================================================
# YOLOX : attributs du fichier d'experience
# ============================================================================
AUG_YOLOX = {
    "mosaic_prob": float(AUG["mosaic"]),
    "mixup_prob": float(AUG["mixup"]),
    "enable_mixup": bool(AUG["mixup"]),
    "hsv_prob": 1.0,
    "flip_prob": AUG["fliplr"],
    "vflip_prob": AUG["flipud"],          # ajoute par notre TrainTransform (patch)
    "degrees": float(AUG["degrees"]),
    "translate": AUG["translate"],
    "mosaic_scale": (1 - AUG["scale"], 1 + AUG["scale"]),
    "shear": float(AUG["shear"]),
    # Gains HSV additifs de YOLOX, regles pour approcher hsv_s / hsv_v = 0.2
    # (0.2 x 255 ~ 51). hgain = 0 car hsv_h = 0.
    "hsv_gains": (0, int(round(AUG["hsv_s"] * 255)), int(round(AUG["hsv_v"] * 255))),
    # Equivalent de close_mosaic=10 d'Ultralytics : mosaique coupee sur les
    # 10 dernieres epoques.
    "no_aug_epochs": 10,
}

# ============================================================================
# Table de correspondance (le "quoi a ete applique a qui", pour l'article)
# ============================================================================
_TABLE = [
    dict(effet="Teinte (hue)", reference="hsv_h = 0",
         ultralytics="hsv_h=0",
         rfdetr="ColorJitter(hue=0)",
         detr="ColorJitter(hue=0)",
         yolox="augment_hsv(hgain=0)",
         ecart="Aucun : teinte inchangee partout."),
    dict(effet="Saturation", reference="hsv_s = 0.2",
         ultralytics="hsv_s=0.2 (gain multiplicatif x[0.8,1.2])",
         rfdetr="ColorJitter(saturation=0.2, p=1)",
         detr="ColorJitter(saturation=0.2)",
         yolox="augment_hsv(sgain=51)",
         ecart="YOLOX applique un decalage ADDITIF (+/-51 sur 255) et non un gain "
               "multiplicatif, et tire l'application de chaque canal a 50%."),
    dict(effet="Luminosite / valeur", reference="hsv_v = 0.2",
         ultralytics="hsv_v=0.2",
         rfdetr="RandomBrightnessContrast(brightness_limit=0.2, p=1)",
         detr="ColorJitter(brightness=0.2)",
         yolox="augment_hsv(vgain=51)",
         ecart="Meme remarque additif/multiplicatif pour YOLOX."),
    dict(effet="Rotation", reference="degrees = 0",
         ultralytics="degrees=0", rfdetr="aucune rotation",
         detr="aucune rotation", yolox="degrees=0.0",
         ecart="Aucun."),
    dict(effet="Translation", reference="translate = 0.1",
         ultralytics="translate=0.1",
         rfdetr="pas de parametre dedie",
         detr="RandomZoomOut + RandomIoUCrop (natifs)",
         yolox="translate=0.1 (random_affine)",
         ecart="RF-DETR et les DETR n'ont pas de translation parametrable : le "
               "recadrage aleatoire natif (IoUCrop/ZoomOut) en tient lieu."),
    dict(effet="Zoom / echelle", reference="scale = 0.25",
         ultralytics="scale=0.25 (facteur [0.75,1.25])",
         rfdetr="resize/crop interne",
         detr="RandomZoomOut + RandomIoUCrop (natifs)",
         yolox="mosaic_scale=(0.75,1.25)",
         ecart="Idem : echelle non parametrable pour RF-DETR et les DETR."),
    dict(effet="Cisaillement / perspective", reference="shear = 0, perspective = 0",
         ultralytics="shear=0, perspective=0", rfdetr="absent",
         detr="absent", yolox="shear=0.0",
         ecart="Aucun."),
    dict(effet="Miroir vertical", reference="flipud = 0.5",
         ultralytics="flipud=0.5",
         rfdetr="VerticalFlip(p=0.5)",
         detr="RandomVerticalFlip(p=0.5) (transform enregistree par le benchmark)",
         yolox="vflip_prob=0.5 (TrainTransform patche par le benchmark)",
         ecart="Aucun dans les faits, mais flipud est ABSENT par defaut de "
               "RT-DETR, D-FINE et YOLOX : il a fallu l'ajouter."),
    dict(effet="Miroir horizontal", reference="fliplr = 0.5",
         ultralytics="fliplr=0.5", rfdetr="HorizontalFlip(p=0.5)",
         detr="RandomHorizontalFlip(p=0.5)", yolox="flip_prob=0.5",
         ecart="Aucun."),
    dict(effet="Mosaique", reference="mosaic = 1",
         ultralytics="mosaic=1.0 (coupee sur les 10 dernieres epoques, close_mosaic)",
         rfdetr="absente du framework",
         detr="absente des deux depots (aucune transform Mosaic dans leur registre)",
         yolox="mosaic_prob=1.0, no_aug_epochs=10",
         ecart="ECART MAJEUR : ni RF-DETR, ni RT-DETR, ni D-FINE ne proposent de "
               "mosaique. Les trois DETR sont entraines sans."),
    dict(effet="MixUp / CutMix", reference="mixup = 0, cutmix = 0",
         ultralytics="mixup=0.0, cutmix=0",
         rfdetr="desactive", detr="desactive",
         yolox="enable_mixup=False, mixup_prob=0.0",
         ecart="Aucun (YOLOX active le mixup par defaut : desactive ici)."),
    dict(effet="Permutation BGR / auto-augment", reference="bgr = 0, auto_augment = None",
         ultralytics="bgr=0.0, auto_augment=None",
         rfdetr="aucun",
         detr="RandomPhotometricDistort REMPLACE par ColorJitter (il permutait les "
              "canaux couleur)",
         yolox="aucun",
         ecart="Le remplacement evite la permutation de canaux, absente de la "
               "reference."),
]


def table():
    """Table de correspondance des augmentations (DataFrame)."""
    return pd.DataFrame(_TABLE)[
        ["effet", "reference", "ultralytics", "rfdetr", "detr", "yolox", "ecart"]
    ]


def write_csv(path=None):
    """Ecrit la table de correspondance a cote des resultats."""
    path = path or cfg.AUG_CSV
    df = table()
    df.to_csv(path, index=False)
    print("Table d'augmentation ecrite ->", path)
    return path
