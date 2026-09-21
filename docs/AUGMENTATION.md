# Augmentation de donnees : reference et equivalences

> Fichier genere par `tools/make_docs.py` depuis `aphids_det/augment.py`.
> Ne pas editer a la main : modifier `augment.py` puis relancer le script.

Regle du benchmark : **les hyperparametres d'entrainement restent ceux par defaut
de chaque modele ; seule l'augmentation est harmonisee.** La reference est le
dictionnaire Ultralytics ci-dessous, applique tel quel a YOLO26n, YOLO11n et
YOLO12n. Les quatre autres modeles recoivent la traduction la plus proche que
leur pipeline permet ; les ecarts qui subsistent sont listes effet par effet.

```python
AUG = {
    'hsv_h': 0,
    'hsv_s': 0.2,
    'hsv_v': 0.2,
    'degrees': 0,
    'translate': 0.1,
    'scale': 0.25,
    'shear': 0,
    'perspective': 0,
    'flipud': 0.5,
    'fliplr': 0.5,
    'mosaic': 1,
    'mixup': 0.0,
    'cutmix': 0,
    'bgr': 0.0,
    'auto_augment': None,
}
```

Les deux ajouts de code faits par le benchmark aux depots externes :

| Depot | Ajout | Pourquoi |
|---|---|---|
| RT-DETR, D-FINE | enregistrement de `RandomVerticalFlip` et `ColorJitter` (`aphids_det/assets/aphids_ops.py`) | aucun miroir vertical dans leurs configs ; `RandomPhotometricDistort` permute aussi les canaux couleur |
| YOLOX | `AphidsTrainTransform` substitue a `TrainTransform` (`aphids_det/assets/yolox_exp_aphids.py`) | ajoute le miroir vertical et fixe les gains HSV |


## Effet par effet

### Teinte (hue)

- **Reference** : `hsv_h = 0`
- **Ultralytics (YOLO26n/11n/12n)** : hsv_h=0
- **RF-DETR-N** : ColorJitter(hue=0)
- **RT-DETR-R18 / D-FINE-N** : ColorJitter(hue=0)
- **YOLOX-Nano** : augment_hsv(hgain=0)
- **Ecart** : Aucun : teinte inchangee partout.

### Saturation

- **Reference** : `hsv_s = 0.2`
- **Ultralytics (YOLO26n/11n/12n)** : hsv_s=0.2 (gain multiplicatif x[0.8,1.2])
- **RF-DETR-N** : ColorJitter(saturation=0.2, p=1)
- **RT-DETR-R18 / D-FINE-N** : ColorJitter(saturation=0.2)
- **YOLOX-Nano** : augment_hsv(sgain=51)
- **Ecart** : YOLOX applique un decalage ADDITIF (+/-51 sur 255) et non un gain multiplicatif, et tire l'application de chaque canal a 50%.

### Luminosite / valeur

- **Reference** : `hsv_v = 0.2`
- **Ultralytics (YOLO26n/11n/12n)** : hsv_v=0.2
- **RF-DETR-N** : RandomBrightnessContrast(brightness_limit=0.2, p=1)
- **RT-DETR-R18 / D-FINE-N** : ColorJitter(brightness=0.2)
- **YOLOX-Nano** : augment_hsv(vgain=51)
- **Ecart** : Meme remarque additif/multiplicatif pour YOLOX.

### Rotation

- **Reference** : `degrees = 0`
- **Ultralytics (YOLO26n/11n/12n)** : degrees=0
- **RF-DETR-N** : aucune rotation
- **RT-DETR-R18 / D-FINE-N** : aucune rotation
- **YOLOX-Nano** : degrees=0.0
- **Ecart** : Aucun.

### Translation

- **Reference** : `translate = 0.1`
- **Ultralytics (YOLO26n/11n/12n)** : translate=0.1
- **RF-DETR-N** : pas de parametre dedie
- **RT-DETR-R18 / D-FINE-N** : RandomZoomOut + RandomIoUCrop (natifs)
- **YOLOX-Nano** : translate=0.1 (random_affine)
- **Ecart** : RF-DETR et les DETR n'ont pas de translation parametrable : le recadrage aleatoire natif (IoUCrop/ZoomOut) en tient lieu.

### Zoom / echelle

- **Reference** : `scale = 0.25`
- **Ultralytics (YOLO26n/11n/12n)** : scale=0.25 (facteur [0.75,1.25])
- **RF-DETR-N** : resize/crop interne
- **RT-DETR-R18 / D-FINE-N** : RandomZoomOut + RandomIoUCrop (natifs)
- **YOLOX-Nano** : mosaic_scale=(0.75,1.25)
- **Ecart** : Idem : echelle non parametrable pour RF-DETR et les DETR.

### Cisaillement / perspective

- **Reference** : `shear = 0, perspective = 0`
- **Ultralytics (YOLO26n/11n/12n)** : shear=0, perspective=0
- **RF-DETR-N** : absent
- **RT-DETR-R18 / D-FINE-N** : absent
- **YOLOX-Nano** : shear=0.0
- **Ecart** : Aucun.

### Miroir vertical

- **Reference** : `flipud = 0.5`
- **Ultralytics (YOLO26n/11n/12n)** : flipud=0.5
- **RF-DETR-N** : VerticalFlip(p=0.5)
- **RT-DETR-R18 / D-FINE-N** : RandomVerticalFlip(p=0.5) (transform enregistree par le benchmark)
- **YOLOX-Nano** : vflip_prob=0.5 (TrainTransform patche par le benchmark)
- **Ecart** : Aucun dans les faits, mais flipud est ABSENT par defaut de RT-DETR, D-FINE et YOLOX : il a fallu l'ajouter.

### Miroir horizontal

- **Reference** : `fliplr = 0.5`
- **Ultralytics (YOLO26n/11n/12n)** : fliplr=0.5
- **RF-DETR-N** : HorizontalFlip(p=0.5)
- **RT-DETR-R18 / D-FINE-N** : RandomHorizontalFlip(p=0.5)
- **YOLOX-Nano** : flip_prob=0.5
- **Ecart** : Aucun.

### Mosaique

- **Reference** : `mosaic = 1`
- **Ultralytics (YOLO26n/11n/12n)** : mosaic=1.0 (coupee sur les 10 dernieres epoques, close_mosaic)
- **RF-DETR-N** : absente du framework
- **RT-DETR-R18 / D-FINE-N** : absente des deux depots (aucune transform Mosaic dans leur registre)
- **YOLOX-Nano** : mosaic_prob=1.0, no_aug_epochs=10
- **Ecart** : ECART MAJEUR : ni RF-DETR, ni RT-DETR, ni D-FINE ne proposent de mosaique. Les trois DETR sont entraines sans.

### MixUp / CutMix

- **Reference** : `mixup = 0, cutmix = 0`
- **Ultralytics (YOLO26n/11n/12n)** : mixup=0.0, cutmix=0
- **RF-DETR-N** : desactive
- **RT-DETR-R18 / D-FINE-N** : desactive
- **YOLOX-Nano** : enable_mixup=False, mixup_prob=0.0
- **Ecart** : Aucun (YOLOX active le mixup par defaut : desactive ici).

### Permutation BGR / auto-augment

- **Reference** : `bgr = 0, auto_augment = None`
- **Ultralytics (YOLO26n/11n/12n)** : bgr=0.0, auto_augment=None
- **RF-DETR-N** : aucun
- **RT-DETR-R18 / D-FINE-N** : RandomPhotometricDistort REMPLACE par ColorJitter (il permutait les canaux couleur)
- **YOLOX-Nano** : aucun
- **Ecart** : Le remplacement evite la permutation de canaux, absente de la reference.


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
AUG_RFDETR = {
    'HorizontalFlip': {'p': 0.5},
    'VerticalFlip': {'p': 0.5},
    'RandomBrightnessContrast': {'brightness_limit': 0.2, 'contrast_limit': 0.0, 'p': 1.0},
    'ColorJitter': {'brightness': 0.0, 'contrast': 0.0, 'saturation': 0.2, 'hue': 0, 'p': 1.0},
}
```

### YOLOX (attributs du fichier d'experience)

```python
AUG_YOLOX = {
    'mosaic_prob': 1.0,
    'mixup_prob': 0.0,
    'enable_mixup': False,
    'hsv_prob': 1.0,
    'flip_prob': 0.5,
    'vflip_prob': 0.5,
    'degrees': 0.0,
    'translate': 0.1,
    'mosaic_scale': (0.75, 1.25),
    'shear': 0.0,
    'hsv_gains': (0, 51, 51),
    'no_aug_epochs': 10,
}
```

### RT-DETR et D-FINE

La liste de transforms de la config d'origine du depot est lue puis reecrite par
`aphids_det/runners/detr_repo.py::patch_ops` :

- `RandomPhotometricDistort` devient
  `{type: ColorJitter, brightness: 0.2, contrast: 0.0, saturation: 0.2, hue: 0.0}` ;
- `RandomVerticalFlip(p=0.5)` est insere juste apres `RandomHorizontalFlip(p=0.5)` ;
- tout le reste (`RandomZoomOut`, `RandomIoUCrop`, `Sanitize...`, `Resize`, les
  conversions de fin de pipeline) est conserve tel quel.

Cette reecriture part de la config lue dans le depot clone : elle suit donc
automatiquement les differences entre RT-DETRv2 et D-FINE (noms des transforms
de fin de pipeline, presence d'une politique `stop_epoch`, cle de batch).
