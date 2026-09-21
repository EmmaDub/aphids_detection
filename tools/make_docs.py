# -*- coding: utf-8 -*-
"""Genere docs/AUGMENTATION.md depuis aphids_det/augment.py (source unique).

    python tools/make_docs.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from aphids_det.augment import AUG, AUG_RFDETR, AUG_YOLOX, _TABLE  # noqa: E402

DOC = Path(__file__).resolve().parent.parent / "docs" / "AUGMENTATION.md"

HEADER = """# Augmentation de donnees : reference et equivalences

> Fichier genere par `tools/make_docs.py` depuis `aphids_det/augment.py`.
> Ne pas editer a la main : modifier `augment.py` puis relancer le script.

Regle du benchmark : **les hyperparametres d'entrainement restent ceux par defaut
de chaque modele ; seule l'augmentation est harmonisee.** La reference est le
dictionnaire Ultralytics ci-dessous, applique tel quel a YOLO26n, YOLO11n et
YOLO12n. Les quatre autres modeles recoivent la traduction la plus proche que
leur pipeline permet ; les ecarts qui subsistent sont listes effet par effet.

```python
AUG = {aug}
```

Les deux ajouts de code faits par le benchmark aux depots externes :

| Depot | Ajout | Pourquoi |
|---|---|---|
| RT-DETR, D-FINE | enregistrement de `RandomVerticalFlip` et `ColorJitter` (`aphids_det/assets/aphids_ops.py`) | aucun miroir vertical dans leurs configs ; `RandomPhotometricDistort` permute aussi les canaux couleur |
| YOLOX | `AphidsTrainTransform` substitue a `TrainTransform` (`aphids_det/assets/yolox_exp_aphids.py`) | ajoute le miroir vertical et fixe les gains HSV |

"""

FOOTER = """
## Voir le resultat

`notebooks/07_visualisation_augmentations.ipynb` affiche cote a cote ce que
chaque pipeline produit reellement sur une meme tuile (code de chaque framework
execute, pas une imitation). Le tableau ci-dessous se verifie a l'oeil sur cette
figure.

## Ce qui n'a pas pu etre harmonise

1. **Mosaique.** Ni RF-DETR, ni RT-DETR, ni D-FINE ne proposent de mosaique
   (verifie dans leurs registres de transforms). Les YOLO et YOLOX s'entrainent
   donc avec `mosaic = 1`, les trois DETR sans. C'est l'ecart le plus lourd de
   consequences : a noter dans toute publication des resultats.
2. **Translation et echelle.** `translate = 0.1` et `scale = 0.25` n'ont pas
   d'equivalent parametrable chez RF-DETR et les DETR. Leurs recadrages natifs
   (`RandomIoUCrop`, `RandomZoomOut`) jouent ce role, avec des amplitudes qui
   leur sont propres.
3. **Forme de la perturbation HSV.** YOLO applique un gain multiplicatif
   (x[0.8, 1.2]) ; YOLOX un decalage additif (+/-51 sur 255) et tire de plus
   l'application de chaque canal a pile ou face. Les amplitudes sont du meme
   ordre, la loi ne l'est pas.
4. **Coupure des augmentations en fin d'entrainement.** Ultralytics coupe la
   mosaique sur les 10 dernieres epoques (`close_mosaic = 10`). Le benchmark
   aligne YOLOX (`no_aug_epochs = 10`) et les DETR (politique `stop_epoch` a
   l'epoque 20 sur 30) sur cette convention.

## Parametres effectivement passes

### RF-DETR (`aug_config`)

```python
AUG_RFDETR = {rfdetr}
```

### YOLOX (attributs du fichier d'experience)

```python
AUG_YOLOX = {yolox}
```

### RT-DETR et D-FINE

La liste de transforms de la config d'origine du depot est lue puis reecrite par
`aphids_det/runners/detr_repo.py::patch_ops` :

- `RandomPhotometricDistort` devient
  `{{type: ColorJitter, brightness: 0.2, contrast: 0.0, saturation: 0.2, hue: 0.0}}` ;
- `RandomVerticalFlip(p=0.5)` est insere juste apres `RandomHorizontalFlip(p=0.5)` ;
- tout le reste (`RandomZoomOut`, `RandomIoUCrop`, `Sanitize...`, `Resize`, les
  conversions de fin de pipeline) est conserve tel quel.

Cette reecriture part de la config lue dans le depot clone : elle suit donc
automatiquement les differences entre RT-DETRv2 et D-FINE (noms des transforms
de fin de pipeline, presence d'une politique `stop_epoch`, cle de batch).
"""


def fmt_dict(d, indent=4):
    lines = ["{"]
    for k, v in d.items():
        lines.append(" " * indent + f"{k!r}: {v!r},")
    lines.append("}")
    return "\n".join(lines)


def main():
    parts = [HEADER.format(aug=fmt_dict(AUG))]
    parts.append("## Effet par effet\n")
    for row in _TABLE:
        parts.append(f"### {row['effet']}\n")
        parts.append(f"- **Reference** : `{row['reference']}`")
        parts.append(f"- **Ultralytics (YOLO26n/11n/12n)** : {row['ultralytics']}")
        parts.append(f"- **RF-DETR-N** : {row['rfdetr']}")
        parts.append(f"- **RT-DETR-R18 / D-FINE-N** : {row['detr']}")
        parts.append(f"- **YOLOX-Nano** : {row['yolox']}")
        parts.append(f"- **Ecart** : {row['ecart']}\n")
    parts.append(FOOTER.format(rfdetr=fmt_dict(AUG_RFDETR), yolox=fmt_dict(AUG_YOLOX)))

    DOC.parent.mkdir(parents=True, exist_ok=True)
    DOC.write_text("\n".join(parts), encoding="utf-8")
    print("ecrit :", DOC)


if __name__ == "__main__":
    main()
