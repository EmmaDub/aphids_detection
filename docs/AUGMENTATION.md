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

Les effets nommes par la reference, et ce que chaque depot en fait.

### Teinte (hue)

- **Reference** : `hsv_h = 0`
- **YOLO26n / YOLO11n / YOLO12n** : hsv_h=0
- **RF-DETR-N** : ColorJitter(hue=0)
- **RT-DETR-R18 / D-FINE-N** : ColorJitter(hue=0)
- **YOLOX-Nano** : augment_hsv(hgain=0)
- **Ecart** : Aucun : teinte inchangee partout.

### Saturation

- **Reference** : `hsv_s = 0.2`
- **YOLO26n / YOLO11n / YOLO12n** : hsv_s=0.2 (gain multiplicatif x[0.8,1.2])
- **RF-DETR-N** : ColorJitter(saturation=0.2, p=1)
- **RT-DETR-R18 / D-FINE-N** : ColorJitter(saturation=0.2)
- **YOLOX-Nano** : augment_hsv(sgain=51)
- **Ecart** : YOLOX applique un decalage ADDITIF (+/-51 sur 255) et non un gain multiplicatif, et tire l'application de chaque canal a 50%.

### Luminosite / valeur

- **Reference** : `hsv_v = 0.2`
- **YOLO26n / YOLO11n / YOLO12n** : hsv_v=0.2
- **RF-DETR-N** : RandomBrightnessContrast(brightness_limit=0.2, p=1)
- **RT-DETR-R18 / D-FINE-N** : ColorJitter(brightness=0.2)
- **YOLOX-Nano** : augment_hsv(vgain=51)
- **Ecart** : Meme remarque additif/multiplicatif pour YOLOX.

### Rotation

- **Reference** : `degrees = 0`
- **YOLO26n / YOLO11n / YOLO12n** : degrees=0
- **RF-DETR-N** : aucune rotation
- **RT-DETR-R18 / D-FINE-N** : aucune rotation
- **YOLOX-Nano** : degrees=0.0
- **Ecart** : Aucun.

### Translation

- **Reference** : `translate = 0.1`
- **YOLO26n / YOLO11n / YOLO12n** : translate=0.1
- **RF-DETR-N** : position aleatoire du recadrage interne (branche B du OneOf)
- **RT-DETR-R18 / D-FINE-N** : position aleatoire du RandomIoUCrop (recadre 80-100% du cote)
- **YOLOX-Nano** : translate=0.1 (random_affine)
- **Ecart** : Aucun des trois DETR n'expose de translation : elle resulte du tirage de position de leur recadrage interne (borne par min_scale=0.8 chez RT-DETR et D-FINE, non reglable chez RF-DETR). Amplitude du meme ordre que translate=0.1, loi differente.

### Zoom / echelle

- **Reference** : `scale = 0.25 (facteur [0.75,1.25])`
- **YOLO26n / YOLO11n / YOLO12n** : scale=0.25
- **RF-DETR-N** : OneOf : resize direct, ou resize 400/500/600 + recadrage + resize 640 -- variation d'echelle reelle mais non parametrable
- **RT-DETR-R18 / D-FINE-N** : RandomZoomOut(side_range=(1.0,1.333)) -> x[0.75,1.0] et RandomIoUCrop(min_scale=0.8) -> x[1.0,1.25]
- **YOLOX-Nano** : mosaic_scale=(0.75,1.25)
- **Ecart** : Bornes des transforms natives recalees sur la reference (cfg.DETR_GEOM='reference'). Par defaut les depots tirent dans x[0.25,3.3], sans commune mesure avec la reference ; cfg.DETR_GEOM='natif' restaure ce comportement, 'affine' remplace les deux ops par un RandomAffine aux parametres exacts d'Ultralytics. RF-DETR reste sans zoom parametrable.

### Cisaillement / perspective

- **Reference** : `shear = 0, perspective = 0`
- **YOLO26n / YOLO11n / YOLO12n** : shear=0, perspective=0
- **RF-DETR-N** : absent
- **RT-DETR-R18 / D-FINE-N** : absent
- **YOLOX-Nano** : shear=0.0
- **Ecart** : Aucun.

### Miroir vertical

- **Reference** : `flipud = 0.5`
- **YOLO26n / YOLO11n / YOLO12n** : flipud=0.5
- **RF-DETR-N** : VerticalFlip(p=0.5)
- **RT-DETR-R18 / D-FINE-N** : RandomVerticalFlip(p=0.5) (transform enregistree par le benchmark)
- **YOLOX-Nano** : vflip_prob=0.5 (TrainTransform patche par le benchmark)
- **Ecart** : Aucun dans les faits, mais flipud est ABSENT par defaut de RT-DETR, D-FINE et YOLOX : il a fallu l'ajouter.

### Miroir horizontal

- **Reference** : `fliplr = 0.5`
- **YOLO26n / YOLO11n / YOLO12n** : fliplr=0.5
- **RF-DETR-N** : HorizontalFlip(p=0.5)
- **RT-DETR-R18 / D-FINE-N** : RandomHorizontalFlip(p=0.5)
- **YOLOX-Nano** : flip_prob=0.5
- **Ecart** : Aucun.

### Mosaique

- **Reference** : `mosaic = 1`
- **YOLO26n / YOLO11n / YOLO12n** : mosaic=1.0 (coupee sur les 10 dernieres epoques, close_mosaic)
- **RF-DETR-N** : absente du framework
- **RT-DETR-R18 / D-FINE-N** : absente des deux depots (aucune transform Mosaic dans leur registre)
- **YOLOX-Nano** : mosaic_prob=1.0, no_aug_epochs=10
- **Ecart** : ECART MAJEUR : ni RF-DETR, ni RT-DETR, ni D-FINE ne proposent de mosaique. Les trois DETR sont entraines sans.

### MixUp / CutMix

- **Reference** : `mixup = 0, cutmix = 0`
- **YOLO26n / YOLO11n / YOLO12n** : mixup=0.0, cutmix=0
- **RF-DETR-N** : desactive
- **RT-DETR-R18 / D-FINE-N** : desactive
- **YOLOX-Nano** : enable_mixup=False, mixup_prob=0.0
- **Ecart** : Aucun (YOLOX active le mixup par defaut : desactive ici).

### Boites tronquees par l'augmentation

- **Reference** : `conserver >= 20% de l'aire d'origine`
- **YOLO26n / YOLO11n / YOLO12n** : box_candidates(area_thr=0.2) -- le 0.10 code en dur est patche par le benchmark
- **RF-DETR-N** : min_visibility=0.2 impose au BboxParams du package (defaut 0.0 : un eclat de boite restait annote)
- **RT-DETR-R18 / D-FINE-N** : RandomIoUCrop ne garde que les boites dont le CENTRE tombe dans le recadrage : une boite conservee garde donc au moins 25% de son aire, deja plus strict que le seuil
- **YOLOX-Nano** : filtre ajoute a la sortie de random_affine (seuil 0.2)
- **Ecart** : Regle harmonisee, par quatre mecanismes differents. Attention : chez RT-DETR et D-FINE le critere natif est le CENTRE de la boite, pas son aire -- une boite visible a plus de 20 % mais dont le centre sort du recadrage est supprimee. Ces deux modeles sont donc plus stricts que la regle, jamais plus laxistes.

### Remplissage des bords vides

- **Reference** : `gris 114 (YOLO)`
- **YOLO26n / YOLO11n / YOLO12n** : 114
- **RF-DETR-N** : sans objet
- **RT-DETR-R18 / D-FINE-N** : fill=114 (les depots utilisent du noir, fill=0)
- **YOLOX-Nano** : 114 (borderValue de random_affine)
- **Ecart** : Aucun apres harmonisation.

### Permutation BGR / auto-augment

- **Reference** : `bgr = 0, auto_augment = None`
- **YOLO26n / YOLO11n / YOLO12n** : bgr=0.0, auto_augment=None
- **RF-DETR-N** : aucun
- **RT-DETR-R18 / D-FINE-N** : RandomPhotometricDistort REMPLACE par ColorJitter (il permutait les canaux couleur)
- **YOLOX-Nano** : aucun
- **Ecart** : Le remplacement evite la permutation de canaux, absente de la reference.

## Augmentations absentes de la reference

Ce que certains depots appliquent EN PLUS, sans qu'aucune cle du dict `AUG` en parle. Ce sont les ecarts les plus faciles a manquer : ils ne figurent dans aucun reglage, et pourtant ils changent ce que le modele voit.

### Transforms albumentations d'Ultralytics

- **Reference** : `absent`
- **YOLO26n / YOLO11n / YOLO12n** : Blur, MedianBlur, ToGray et CLAHE, a p=0.01 chacun, appliques SI le paquet albumentations est importable
- **RF-DETR-N** : aucun equivalent
- **RT-DETR-R18 / D-FINE-N** : aucun equivalent
- **YOLOX-Nano** : aucun equivalent
- **Ecart** : Environ 4 % des tuiles recoivent un flou, un passage en niveaux de gris ou une egalisation d'histogramme -- chez les YOLO seulement. Le bloc est silencieux : il s'active selon la presence d'albumentations dans le runtime. augment.etat_albumentations() le signale et la colonne notes du CSV enregistre ce qui s'est reellement passe.

### Multi-echelle par lot

- **Reference** : `absent : imgsz fixe a 640`
- **YOLO26n / YOLO11n / YOLO12n** : multi_scale = False par defaut en detection
- **RF-DETR-N** : multi_scale et expanded_scales sont a True par defaut : la resolution change d'un lot a l'autre. DESACTIVES par le benchmark (repli silencieux si la version les refuse)
- **RT-DETR-R18 / D-FINE-N** : BatchImageCollateFunction neutralise par les configs retenues (scales: ~ pour RT-DETRv2-R18, base_size_repeat: ~ pour D-FINE-N)
- **YOLOX-Nano** : multiscale_range mis a 0 par le benchmark (defaut 5, soit +/-160 px autour de 640)
- **Ecart** : Sans cette harmonisation, RF-DETR aurait ete le seul a voir plusieurs resolutions, et YOLOX le seul autre a varier de +/-160 px.

### Coupure des augmentations en fin d'entrainement

- **Reference** : `close_mosaic = 10 (defaut Ultralytics)`
- **YOLO26n / YOLO11n / YOLO12n** : close_mosaic=10 : mosaique coupee sur les 10 dernieres epoques
- **RF-DETR-N** : aucun mecanisme de ce type
- **RT-DETR-R18 / D-FINE-N** : politique stop_epoch : ColorJitter, ZoomOut et IoUCrop coupes a partir de l'epoque 20 sur 30
- **YOLOX-Nano** : no_aug_epochs=10
- **Ecart** : Aligne sur les 10 dernieres epoques partout, sauf RF-DETR qui n'offre pas ce reglage.

### Plage des pixels et normalisation

- **Reference** : `non specifie (pretraitement, pas augmentation)`
- **YOLO26n / YOLO11n / YOLO12n** : 0-1, RGB
- **RF-DETR-N** : 0-1 puis normalisation ImageNet (mean/std)
- **RT-DETR-R18 / D-FINE-N** : 0-1, sans normalisation ImageNet (ConvertPILImage scale=True)
- **YOLOX-Nano** : 0-255 bruts, BGR, sans normalisation
- **Ecart** : Non harmonise, et il ne FAUT pas l'harmoniser : chaque depot doit garder le pretraitement de ses poids COCO, sans quoi le transfert est casse.

### Mise a 640 de la tuile

- **Reference** : `imgsz = 640`
- **YOLO26n / YOLO11n / YOLO12n** : letterbox, ratio preserve, remplissage 114
- **RF-DETR-N** : redimensionnement carre
- **RT-DETR-R18 / D-FINE-N** : Resize [640, 640]
- **YOLOX-Nano** : letterbox, ratio preserve, remplissage 114
- **Ecart** : Sans consequence ici : les tuiles sont deja carrees, 640 x 640.

### Effacement aleatoire et copier-coller d'instances

- **Reference** : `absents`
- **YOLO26n / YOLO11n / YOLO12n** : erasing=0.4 mais classification uniquement ; copy_paste=0.0 et exige des masques de segmentation
- **RF-DETR-N** : absents
- **RT-DETR-R18 / D-FINE-N** : absents
- **YOLOX-Nano** : absents
- **Ecart** : Aucun effet en detection : ces deux reglages ne sont jamais atteints par le pipeline utilise ici.


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
2. **Translation et echelle chez RF-DETR.** RF-DETR n'expose ni translation ni
   zoom : son redimensionnement interne est le seul effet d'echelle, et il n'est
   pas reglable depuis `aug_config`. Pour RT-DETR et D-FINE en revanche, les
   bornes de `RandomZoomOut` et `RandomIoUCrop` sont desormais calees sur la
   reference (`cfg.DETR_GEOM = "reference"`), ce qui ramene leur echelle a
   x[0.75, 1.25] : par defaut ces deux ops tiraient dans x[0.25, 3.3], avec une
   deformation du rapport d'aspect jusqu'a 2:1 et un remplissage noir.
3. **Forme de la perturbation HSV.** YOLO applique un gain multiplicatif
   (x[0.8, 1.2]) ; YOLOX un decalage additif (+/-51 sur 255) et tire de plus
   l'application de chaque canal a pile ou face. Les amplitudes sont du meme
   ordre, la loi ne l'est pas.
4. **Forme de la translation chez les DETR.** Elle n'est pas parametrable : elle
   resulte du tirage de position du recadrage `RandomIoUCrop`, borne par
   `min_scale = 0.8`. L'amplitude est du meme ordre que `translate = 0.1`, la loi
   ne l'est pas.
5. **Coupure des augmentations en fin d'entrainement.** Ultralytics coupe la
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
