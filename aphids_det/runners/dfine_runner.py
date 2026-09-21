# -*- coding: utf-8 -*-
"""D-FINE-N via le depot officiel Peterande/D-FINE.

Config de base : `configs/dfine/dfine_hgnetv2_n_coco.yml` (variante nano,
backbone HGNetv2-B0), poids COCO `dfine_n_coco.pth` passes en `--tuning`.
Le depot partage la base de code de RT-DETRv2 : tout passe par detr_repo.
"""

from . import detr_repo

VARIANT = "dfine_n"


def run_cv(folds=None, install=True):
    """Validation croisee complete de D-FINE-N."""
    return detr_repo.run_cv(VARIANT, folds=folds, install=install)


def run_fold(fold, install=True):
    """Un seul fold (utile pour un test rapide)."""
    return detr_repo.run_fold(VARIANT, fold, install=install)
