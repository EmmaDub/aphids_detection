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
# Libelles des modeles : une seule source, pour que la table d'augmentation,
# la documentation et les figures designent les modeles de la meme facon.
# ============================================================================
MODELES = {
    "ultralytics": "YOLO26n / YOLO11n / YOLO12n",
    "rfdetr": "RF-DETR-N",
    "detr": "RT-DETR-R18 / D-FINE-N",
    "yolox": "YOLOX-Nano",
}

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
    # Remplace le RandomPhotometricDistort natif : retire d'un coup la teinte
    # et la permutation de canaux, absentes de la reference.
    "ColorJitter": {"brightness": AUG["hsv_v"], "contrast": 0.0,
                    "saturation": AUG["hsv_s"], "hue": AUG["hsv_h"], "p": 1.0},
    # Seule source d'echelle et de translation, bornee sur la reference : les
    # couches natives (scale_jitter, multi_scale) sont coupees a l'appel de
    # train() dans runners/rfdetr_runner.py.
    # keep_ratio=True : sans lui albumentations tire les facteurs x et y
    # independamment, la ou Ultralytics applique un facteur isotrope.
    # fill=114 : meme gris que YOLO et YOLOX pour les bords decouverts
    # (parametre nomme `fill` depuis albumentations 2.x, `cval` avant).
    "Affine": {"scale": (1 - AUG["scale"], 1 + AUG["scale"]),
               "translate_percent": (-AUG["translate"], AUG["translate"]),
               "rotate": (AUG["degrees"], AUG["degrees"]),
               "shear": (AUG["shear"], AUG["shear"]),
               "keep_ratio": True, "fill": 114, "p": 1.0},
}

# Couches d'echelle natives de RF-DETR, coupees pour que l'Affine ci-dessus
# soit le seul mecanisme d'echelle (cf. docs/AUGMENTATION.md).
RFDETR_COUPURES = {"scale_jitter": False, "multi_scale": False,
                   "expanded_scales": False}

# ============================================================================
# RT-DETR / D-FINE : ops inserees dans la liste de transforms du depot
# (le reste de la liste d'origine est conserve tel quel, cf. runners/detr_repo.py)
# ============================================================================
DETR_COLOR_OP = {"type": "ColorJitter",
                 "brightness": AUG["hsv_v"], "contrast": 0.0,
                 "saturation": AUG["hsv_s"], "hue": AUG["hsv_h"]}
DETR_HFLIP_OP = {"type": "RandomHorizontalFlip", "p": AUG["fliplr"]}
DETR_VFLIP_OP = {"type": "RandomVerticalFlip", "p": AUG["flipud"]}

# --- Geometrie calee sur la reference (cfg.DETR_GEOM == "reference") --------
# Par defaut les depots tirent une echelle dans x[0.25, 3.3] : RandomZoomOut
# agrandit la toile jusqu'a x4 (soit un objet reduit a 1/4) et RandomIoUCrop
# recadre jusqu'a 30% du cote avant retour a 640 (soit un objet grossi x3.3).
# Les bornes ci-dessous ramenent l'ensemble a x[1-scale, 1+scale], comme YOLO.
_ZOOM_OUT_MAX = round(1.0 / (1.0 - AUG["scale"]), 3)     # 1.333 -> objet x0.75
_CROP_MIN = round(1.0 / (1.0 + AUG["scale"]), 3)         # 0.8   -> objet x1.25
DETR_ZOOMOUT_OP = {"type": "RandomZoomOut", "fill": 114,
                   "side_range": [1.0, _ZOOM_OUT_MAX], "p": 0.5}
DETR_IOUCROP_OP = {"type": "RandomIoUCrop", "min_scale": _CROP_MIN, "max_scale": 1.0,
                   "min_aspect_ratio": 0.9, "max_aspect_ratio": 1.1, "p": 0.8}
# Variante stricte (cfg.DETR_GEOM == "affine") : parametres d'Ultralytics tels quels.
DETR_AFFINE_OP = {"type": "RandomAffine", "degrees": AUG["degrees"],
                  "translate": [AUG["translate"], AUG["translate"]],
                  "scale": [1 - AUG["scale"], 1 + AUG["scale"]],
                  "shear": AUG["shear"], "fill": 114}

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
    "perspective": float(AUG["perspective"]),
    # Gain HSV MULTIPLICATIF x[0.8, 1.2], comme la reference : le decalage
    # additif natif de YOLOX est remplace dans assets/yolox_exp_aphids.py.
    "hsv_gain": AUG["hsv_s"],
    # Coupure d'augmentation de fin SUPPRIMEE : no_aug_epochs n'a d'equivalent
    # que chez Ultralytics (close_mosaic). La garder donnerait a deux frameworks
    # sur quatre une phase finale sans mosaique. L'augmentation reste donc
    # constante sur tout le budget, pour les sept modeles.
    "no_aug_epochs": 0,
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
         yolox="gain multiplicatif x[0.8,1.2], applique systematiquement "
               "(augment_hsv refait par le benchmark)",
         ecart="Aucun depuis la reecriture : le decalage ADDITIF natif de YOLOX "
               "(+/-30 sur 255) et son tirage a pile ou face par canal, qui ne "
               "saturait qu'une image sur deux, ont ete remplaces."),
    dict(effet="Luminosite / valeur", reference="hsv_v = 0.2",
         ultralytics="hsv_v=0.2",
         rfdetr="ColorJitter(brightness=0.2, p=1)",
         detr="ColorJitter(brightness=0.2)",
         yolox="gain multiplicatif x[0.8,1.2], applique systematiquement",
         ecart="Aucun depuis la reecriture de augment_hsv."),
    dict(effet="Rotation", reference="degrees = 0",
         ultralytics="degrees=0", rfdetr="aucune rotation",
         detr="aucune rotation", yolox="degrees=0.0",
         ecart="Aucun."),
    dict(effet="Translation", reference="translate = 0.1",
         ultralytics="translate=0.1",
         rfdetr="Affine(translate_percent=(-0.1, 0.1))",
         detr="position aleatoire du RandomIoUCrop (recadre 80-100% du cote)",
         yolox="translate=0.1 (random_affine)",
         ecart="RF-DETR a desormais une translation bornee explicitement, par "
               "son Affine. Chez RT-DETR et D-FINE elle reste non parametrable : "
               "elle resulte du tirage de position de RandomIoUCrop, borne par "
               "min_scale=0.8. Amplitude du meme ordre que translate=0.1, loi "
               "differente."),
    dict(effet="Zoom / echelle", reference="scale = 0.25 (facteur [0.75,1.25])",
         ultralytics="scale=0.25",
         rfdetr="Affine(scale=(0.75, 1.25), keep_ratio=True) ; scale_jitter, "
                "multi_scale et expanded_scales coupes",
         detr="RandomZoomOut(side_range=(1.0,1.333)) -> x[0.75,1.0] et "
              "RandomIoUCrop(min_scale=0.8) -> x[1.0,1.25]",
         yolox="mosaic_scale=(0.75,1.25)",
         ecart="Bornes des transforms natives recalees sur la reference "
               "(cfg.DETR_GEOM='reference'). Par defaut les depots tirent dans "
               "x[0.25,3.3], sans commune mesure avec la reference ; "
               "cfg.DETR_GEOM='natif' restaure ce comportement, 'affine' "
               "remplace les deux ops par un RandomAffine aux parametres exacts "
               "d'Ultralytics. RF-DETR est desormais borne lui aussi, par son "
               "Affine, ses trois couches d'echelle natives etant coupees."),
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
         ultralytics="mosaic=1.0, constante sur tout le run (close_mosaic=0)",
         rfdetr="absente du framework",
         detr="absente des deux depots (aucune transform Mosaic dans leur registre)",
         yolox="mosaic_prob=1.0, constante sur tout le run (no_aug_epochs=0)",
         ecart="ECART MAJEUR : ni RF-DETR, ni RT-DETR, ni D-FINE ne proposent de "
               "mosaique. Les trois DETR sont entraines sans."),
    dict(effet="MixUp / CutMix", reference="mixup = 0, cutmix = 0",
         ultralytics="mixup=0.0, cutmix=0",
         rfdetr="desactive", detr="desactive",
         yolox="enable_mixup=False, mixup_prob=0.0",
         ecart="Aucun (YOLOX active le mixup par defaut : desactive ici)."),
    dict(effet="Boites tronquees par l'augmentation",
         reference=f"conserver >= {int(cfg.MIN_VISIBILITY * 100)}% de l'aire d'origine",
         ultralytics=f"box_candidates(area_thr={cfg.MIN_VISIBILITY}) -- le 0.10 "
                     "code en dur est patche par le benchmark",
         rfdetr=f"min_visibility={cfg.MIN_VISIBILITY} impose au BboxParams du "
                "package (defaut 0.0 : un eclat de boite restait annote)",
         detr="RandomIoUCrop ne garde que les boites dont le CENTRE tombe dans "
              "le recadrage : une boite conservee garde donc au moins 25% de son "
              "aire, deja plus strict que le seuil",
         yolox=f"filtre ajoute a la sortie de random_affine (seuil "
               f"{cfg.MIN_VISIBILITY})",
         ecart="Regle harmonisee, par quatre mecanismes differents. Attention : "
               "chez RT-DETR et D-FINE le critere natif est le CENTRE de la boite, "
               "pas son aire -- une boite visible a plus de 20 % mais dont le "
               "centre sort du recadrage est supprimee. Ces deux modeles sont "
               "donc plus stricts que la regle, jamais plus laxistes."),
    dict(effet="Remplissage des bords vides", reference="gris 114 (YOLO)",
         ultralytics="114", rfdetr="Affine(fill=114)",
         detr="fill=114 (les depots utilisent du noir, fill=0)",
         yolox="114 (borderValue de random_affine)",
         ecart="Aucun apres harmonisation."),
    # ------------------------------------------------------------------
    # Effets ABSENTS de la reference, mais appliques par certains depots.
    # Ce sont eux qui biaisent une comparaison sans qu'on les voie.
    # ------------------------------------------------------------------
    dict(effet="Transforms albumentations d'Ultralytics", hors_reference=True,
         reference="absent",
         ultralytics="Blur, MedianBlur, ToGray et CLAHE NEUTRALISES (p=0) par "
                     "patch_ultralytics_albumentations ; sans ce patch ils "
                     "s'appliquent a p=0.01 des qu'albumentations est importable",
         rfdetr="aucun equivalent", detr="aucun equivalent",
         yolox="aucun equivalent",
         ecart="Resolu. Sans le patch, environ 4 % des tuiles recevaient un "
               "flou, un passage en niveaux de gris ou une egalisation "
               "d'histogramme, chez les YOLO seulement -- et de facon "
               "silencieuse, puisque le bloc s'active selon la presence "
               "d'albumentations dans le runtime."),
    dict(effet="Multi-echelle par lot", hors_reference=True,
         reference="absent : imgsz fixe a 640",
         ultralytics="multi_scale = False par defaut en detection",
         rfdetr="multi_scale, expanded_scales et scale_jitter sont actifs par "
                "defaut : resolution variable et recadrage interne. Les TROIS "
                "sont coupes par le benchmark (repli si la version les refuse)",
         detr="BatchImageCollateFunction neutralise par les configs retenues "
              "(scales: ~ pour RT-DETRv2-R18, base_size_repeat: ~ pour D-FINE-N)",
         yolox="multiscale_range mis a 0 par le benchmark (defaut 5, soit "
               "+/-160 px autour de 640)",
         ecart="Sans cette harmonisation, RF-DETR aurait ete le seul a voir "
               "plusieurs resolutions, et YOLOX le seul autre a varier de "
               "+/-160 px."),
    dict(effet="Coupure des augmentations en fin d'entrainement", hors_reference=True,
         reference="supprimee partout",
         ultralytics="close_mosaic=0 (defaut 10 : desactive)",
         rfdetr="aucun mecanisme de ce type, donc rien a desactiver",
         detr="politique stop_epoch repoussee au-dela du plafond, donc inerte",
         yolox="no_aug_epochs=0 (defaut 15 : desactive)",
         ecart="RESIDU ASSUME. close_mosaic et no_aug_epochs n'existent que "
               "chez deux frameworks sur quatre : les garder aurait donne a YOLO "
               "et YOLOX une phase finale sans mosaique que ni RF-DETR ni les "
               "DETR ne peuvent avoir. Ils sont donc desactives, et les sept "
               "modeles voient la meme augmentation sur tout le budget. YOLO et "
               "YOLOX y perdent le petit gain que leur donnait cette queue "
               "d'entrainement."),
    dict(effet="Plage des pixels et normalisation", hors_reference=True,
         reference="non specifie (pretraitement, pas augmentation)",
         ultralytics="0-1, RGB",
         rfdetr="0-1 puis normalisation ImageNet (mean/std)",
         detr="0-1, sans normalisation ImageNet (ConvertPILImage scale=True)",
         yolox="0-255 bruts, BGR, sans normalisation",
         ecart="Non harmonise, et il ne FAUT pas l'harmoniser : chaque depot "
               "doit garder le pretraitement de ses poids COCO, sans quoi le "
               "transfert est casse."),
    dict(effet="Mise a 640 de la tuile", hors_reference=True,
         reference="imgsz = 640",
         ultralytics="letterbox, ratio preserve, remplissage 114",
         rfdetr="resolution=640 passee a RFDETRNano (defaut 384)",
         detr="Resize [640, 640]",
         yolox="letterbox, ratio preserve, remplissage 114",
         ecart="Sans consequence ici : les tuiles sont deja carrees, 640 x 640."),
    dict(effet="Effacement aleatoire et copier-coller d'instances", hors_reference=True,
         reference="absents",
         ultralytics="erasing=0.4 mais classification uniquement ; "
                     "copy_paste=0.0 et exige des masques de segmentation",
         rfdetr="absents", detr="absents", yolox="absents",
         ecart="Aucun effet en detection : ces deux reglages ne sont jamais "
               "atteints par le pipeline utilise ici."),
    dict(effet="Permutation BGR / auto-augment", reference="bgr = 0, auto_augment = None",
         ultralytics="bgr=0.0, auto_augment=None",
         rfdetr="aucun",
         detr="RandomPhotometricDistort REMPLACE par ColorJitter (il permutait les "
              "canaux couleur)",
         yolox="aucun",
         ecart="Le remplacement evite la permutation de canaux, absente de la "
               "reference."),
]


def table(libelles=True, perimetre=None):
    """Table de correspondance des augmentations (DataFrame).

    Les colonnes portent les memes libelles de modeles que les figures du
    notebook 07 (`libelles=False` garde les cles internes).

    La colonne `perimetre` distingue les effets de la reference de ceux qu'un
    depot applique EN PLUS, absents du dict AUG. `perimetre="reference"` ou
    `"hors reference"` filtre les uns ou les autres.
    """
    df = pd.DataFrame(_TABLE)
    df["perimetre"] = df.get("hors_reference", False).fillna(False).map(
        {True: "hors reference", False: "reference"})
    df = df[["effet", "perimetre", "reference",
             "ultralytics", "rfdetr", "detr", "yolox", "ecart"]]
    if perimetre:
        df = df[df["perimetre"] == perimetre].reset_index(drop=True)
    return df.rename(columns=MODELES) if libelles else df


def etat_albumentations():
    """Dit si le bloc albumentations cache d'Ultralytics va s'appliquer.

    Ultralytics ajoute Blur, MedianBlur, ToGray et CLAHE (p=0.01 chacun) des
    que le paquet albumentations est importable, et se tait sinon. Le resultat
    de l'entrainement depend donc du contenu du runtime : on l'enregistre.
    """
    try:
        import albumentations  # noqa: F401
        return True
    except Exception:
        return False


def masque_visibilite(aires_avant, boxes_apres, min_visibility=None):
    """Masque des boites qui conservent assez de leur aire apres rognage.

    `aires_avant` et `boxes_apres` (xyxy) doivent etre alignes. Le meme calcul
    est recopie dans `assets/yolox_exp_aphids.py`, qui tourne dans un processus
    ou `aphids_det` n'est pas importable : toute correction ici doit y etre
    reportee.
    """
    import numpy as np

    seuil = cfg.MIN_VISIBILITY if min_visibility is None else min_visibility
    aires_avant = np.asarray(aires_avant, dtype=float)
    boxes_apres = np.asarray(boxes_apres, dtype=float).reshape(-1, 4)
    visibles = ((boxes_apres[:, 2] - boxes_apres[:, 0]).clip(0) *
                (boxes_apres[:, 3] - boxes_apres[:, 1]).clip(0))
    return visibles > seuil * np.maximum(aires_avant, 1e-6)


def patch_ultralytics_visibility(min_visibility=None):
    """Porte le seuil de visibilite d'Ultralytics a cfg.MIN_VISIBILITY.

    `RandomPerspective` ne garde une boite que si elle conserve au moins 10 % de
    son aire d'origine (`box_candidates(area_thr=0.10)`, valeur codee en dur dans
    l'appel). Le benchmark aligne ce seuil sur celui impose aux autres
    frameworks. Idempotent : reappeler la fonction ne re-empile pas le patch.
    """
    from ultralytics.data.augment import RandomPerspective

    seuil = cfg.MIN_VISIBILITY if min_visibility is None else min_visibility
    if getattr(RandomPerspective.box_candidates, "_aphids_seuil", None) == seuil:
        return seuil
    origine = getattr(RandomPerspective.box_candidates, "_aphids_origine",
                      RandomPerspective.box_candidates)

    def box_candidates(box1, box2, wh_thr=2, ar_thr=100, area_thr=0.1, eps=1e-16):
        return origine(box1, box2, wh_thr=wh_thr, ar_thr=ar_thr,
                       area_thr=seuil, eps=eps)

    box_candidates._aphids_seuil = seuil
    box_candidates._aphids_origine = origine
    RandomPerspective.box_candidates = staticmethod(box_candidates)
    print(f"  Ultralytics : visibilite minimale des boites portee a {seuil:.0%} "
          f"(defaut du framework : 10%)")
    return seuil


def patch_ultralytics_albumentations():
    """Neutralise le bloc albumentations cache d'Ultralytics.

    `ultralytics.data.augment.Albumentations` ajoute Blur, MedianBlur, ToGray et
    CLAHE (p = 0.01 chacun) des que le paquet albumentations est importable, et
    ne fait rien sinon : le resultat d'un entrainement depend donc du contenu du
    runtime, en silence. On force la transform a None, ce que `__call__` traite
    deja comme "ne rien faire". Idempotent.
    """
    from ultralytics.data.augment import Albumentations

    if getattr(Albumentations.__init__, "_aphids_neutralise", False):
        return True
    origine = Albumentations.__init__

    def __init__(self, *args, **kwargs):
        origine(self, *args, **kwargs)
        self.transform = None            # __call__ renvoie alors les labels tels quels
        self.contains_spatial = False

    __init__._aphids_neutralise = True
    __init__._aphids_origine = origine
    Albumentations.__init__ = __init__
    print("  Ultralytics : bloc albumentations cache neutralise "
          "(Blur, MedianBlur, ToGray, CLAHE a p=0)")
    return True


def patch_rfdetr_visibility(min_visibility=None):
    """Porte le seuil de visibilite de RF-DETR a cfg.MIN_VISIBILITY.

    Le pipeline d'entrainement de RF-DETR tire a pile ou face entre un
    redimensionnement direct et un enchainement redimensionnement -> recadrage ->
    redimensionnement. Ce recadrage tronque des boites, et le wrapper
    albumentations du package fixe `min_visibility=0.0` : une boite reduite a un
    eclat reste annotee.

    Le seuil n'est pas atteignable par `aug_config` : on substitue donc, dans le
    seul module `rfdetr.datasets.transforms`, un `BboxParams` qui impose notre
    valeur. Le patch est verifie avant d'etre conserve ; s'il ne prend pas, la
    fonction previent et ne touche a rien (la difference est alors a lire dans
    docs/AUGMENTATION.md).

    Renvoie le seuil applique, ou None si le patch n'a pas pu etre pose.
    """
    import types

    seuil = cfg.MIN_VISIBILITY if min_visibility is None else min_visibility
    try:
        import albumentations as alb
        from rfdetr.datasets import transforms as rf_transforms
    except Exception as e:
        print(f"  RF-DETR : patch de visibilite impossible ({type(e).__name__}: {e})")
        return None

    courant = getattr(rf_transforms, "alb", None)
    if courant is None:
        print("  RF-DETR : module albumentations introuvable dans "
              "rfdetr.datasets.transforms -> seuil de visibilite inchange (0 %)")
        return None
    if getattr(courant, "_aphids_seuil", None) == seuil:
        return seuil

    base = getattr(courant, "_aphids_origine", courant)

    class BboxParams(base.BboxParams):
        """BboxParams de RF-DETR, avec le seuil de visibilite du benchmark."""

        def __init__(self, *args, **kwargs):
            kwargs["min_visibility"] = seuil
            super().__init__(*args, **kwargs)

    proxy = types.ModuleType("albumentations_aphids")
    proxy.__dict__.update(base.__dict__)
    proxy.BboxParams = BboxParams
    proxy._aphids_seuil = seuil
    proxy._aphids_origine = base

    # Verification : le seuil doit effectivement ressortir du BboxParams patche.
    try:
        essai = proxy.BboxParams(format="pascal_voc", label_fields=["category_ids"])
        if float(getattr(essai, "min_visibility", 0.0)) != float(seuil):
            raise ValueError(f"min_visibility={getattr(essai, 'min_visibility', None)}")
    except Exception as e:
        print(f"  RF-DETR : patch de visibilite refuse par la version installee "
              f"({type(e).__name__}: {e}) -> seuil inchange (0 %)")
        return None

    rf_transforms.alb = proxy
    print(f"  RF-DETR : visibilite minimale des boites portee a {seuil:.0%} "
          f"(defaut du package : 0 %)")
    return seuil


def write_csv(path=None):
    """Ecrit la table de correspondance a cote des resultats."""
    path = path or cfg.AUG_CSV
    df = table()
    df.to_csv(path, index=False)
    print("Table d'augmentation ecrite ->", path)
    return path
