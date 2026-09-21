# -*- coding: utf-8 -*-
"""RT-DETR-R18 via le depot officiel lyuwenyu/RT-DETR.

Deux variantes disponibles :

  "rtdetr_v2" (defaut) : `rtdetrv2_pytorch`, config `rtdetrv2_r18vd_120e_coco`,
      poids COCO `rtdetrv2_r18vd_120e_coco.pth`. C'est la branche maintenue du
      depot, compatible avec les versions recentes de torchvision.

  "rtdetr_v1" : `rtdetr_pytorch`, config `rtdetr_r18vd_6x_coco` (le RT-DETR-R18
      de l'article d'origine). ATTENTION : ce code importe
      `torchvision.datapoints`, supprime de torchvision depuis la 0.17 ; il
      demande donc un environnement avec torchvision <= 0.16, ce qui n'est plus
      le cas de Colab par defaut.

Aucun hyperparametre du depot n'est modifie en dehors du nombre d'epoques, du
batch, du dataset et des augmentations (cf. detr_repo.patch_ops).
"""

from . import detr_repo

VARIANT = "rtdetr_v2"


def run_cv(folds=None, variant=VARIANT, install=True):
    """Validation croisee complete de RT-DETR-R18."""
    return detr_repo.run_cv(variant, folds=folds, install=install)


def run_fold(fold, variant=VARIANT, install=True):
    """Un seul fold (utile pour un test rapide)."""
    return detr_repo.run_fold(variant, fold, install=install)
