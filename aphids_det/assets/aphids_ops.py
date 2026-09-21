# -*- coding: utf-8 -*-
"""Transforms ajoutees aux depots RT-DETR et D-FINE par le benchmark.

Ce fichier est copie dans `src/data/` du depot clone, et importe depuis
`src/data/__init__.py`. Il enregistre dans le registre du depot deux transforms
torchvision v2 absentes de leurs configs :

  - `RandomVerticalFlip` : la reference d'augmentation impose flipud = 0.5, or
    aucun des deux depots ne propose de miroir vertical ;
  - `ColorJitter` : remplace `RandomPhotometricDistort`, qui permute aussi les
    canaux couleur (equivalent de bgr = 1, absent de la reference).

Les deux depots n'enregistrent pas de la meme facon
(`register(cls)` pour RT-DETR v1, `register()(cls)` pour RT-DETRv2 et D-FINE) :
la signature est inspectee pour choisir le bon style.
"""

import inspect

import torchvision.transforms.v2 as T

from src.core import register

try:
    from src.core import GLOBAL_CONFIG
except ImportError:                                  # pragma: no cover
    GLOBAL_CONFIG = {}

# register(cls) -> RT-DETR v1 ; register()(cls) -> RT-DETRv2 / D-FINE
_PARAMS = list(inspect.signature(register).parameters)
_V1_STYLE = bool(_PARAMS) and _PARAMS[0] == "cls"


def _register(cls):
    """Enregistre `cls` si son nom est libre, quel que soit le style du depot."""
    if cls.__name__ in GLOBAL_CONFIG:
        return cls
    out = register(cls) if _V1_STYLE else register()(cls)
    return out if out is not None else cls


RandomVerticalFlip = _register(T.RandomVerticalFlip)
ColorJitter = _register(T.ColorJitter)

__all__ = ["RandomVerticalFlip", "ColorJitter"]
