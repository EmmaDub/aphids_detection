# Benchmark de detection de pucerons — 7 modeles legers, un seul protocole

Comparaison de sept detecteurs "plus petite variante" sur des tuiles 640 px de
pucerons (2 classes : `Apterous_aphid`, `Alate_aphid`), en validation croisee
5 folds, avec **les memes donnees, la meme augmentation et la meme evaluation**
pour tous.

| Modele | Source | Poids initiaux |
|---|---|---|
| YOLO26n | package `ultralytics` | `yolo26n.pt` (COCO) |
| YOLO11n | package `ultralytics` | `yolo11n.pt` (COCO) |
| YOLO12n | package `ultralytics` | `yolo12n.pt` (COCO) |
| RF-DETR Nano | package `rfdetr` (Roboflow) | pre-entraines RF-DETR |
| RT-DETR-R18 | depot officiel [lyuwenyu/RT-DETR](https://github.com/lyuwenyu/RT-DETR) | `rtdetrv2_r18vd_120e_coco.pth` |
| D-FINE-N | depot officiel [Peterande/D-FINE](https://github.com/Peterande/D-FINE) | `dfine_n_coco.pth` |
| YOLOX-Nano | depot officiel [Megvii-BaseDetection/YOLOX](https://github.com/Megvii-BaseDetection/YOLOX) | `yolox_nano.pth` |

## Les trois regles du benchmark

1. **Memes donnees.** La liste des tuiles d'entrainement de chaque fold est
   calculee une fois, figee sur le Drive (`splits_figes/train_fold{f}_neg3.txt`)
   et relue telle quelle par les sept modeles. Le fold 5 est reserve au test et
   n'est jamais vu pendant la comparaison.
2. **Memes augmentations.** Les hyperparametres d'entrainement restent ceux par
   defaut de chaque depot ; seule l'augmentation est harmonisee sur la reference
   Ultralytics. Les equivalences et les ecarts residuels sont documentes effet
   par effet dans [docs/AUGMENTATION.md](docs/AUGMENTATION.md).
3. **Meme evaluation.** Aucune metrique native n'est utilisee. Chaque modele
   exporte ses predictions du fold de validation au format COCO, et un unique
   evaluateur (`aphids_det/evaluate.py`) calcule mAP@0.5, mAP@0.5:0.95, P, R, F1,
   TP, FP, FN. Latence CPU, parametres et taille sont mesures de la meme facon
   pour tous (`aphids_det/bench.py`).

## Environnement

Concu pour **Google Colab** (montage Drive, `/content`, GPU T4). Les quatre
frameworks ont des dependances incompatibles entre elles : il y a donc **un
notebook par framework**, chacun dans son propre runtime.

## Demarrage

```text
notebooks/00_preparation_folds.ipynb   <- a lancer en premier, une fois
notebooks/01_ultralytics.ipynb         YOLO26n + YOLO11n + YOLO12n
notebooks/02_rfdetr.ipynb              RF-DETR Nano
notebooks/03_rtdetr_r18.ipynb          RT-DETR-R18
notebooks/04_dfine_n.ipynb             D-FINE-N
notebooks/05_yolox_nano.ipynb          YOLOX-Nano
notebooks/06_synthese.ipynb            <- moyennes + classeur Excel

notebooks/07_visualisation_augmentations.ipynb   <- voir les augmentations
                                                    (independant, sans GPU)
```

Chaque notebook clone ce depot dans `/content/aphids_detection`, monte le Drive,
reconstruit les folds (liens symboliques locaux, a refaire a chaque session),
puis lance sa validation croisee. **La reprise est automatique** : un couple
(modele, fold) deja present dans le CSV est saute, et une erreur sur un fold
n'interrompt pas les suivants.

**Depot prive et Colab.** Ce depot etant prive, `git clone` en HTTPS echoue dans
Colab avec `could not read Username for 'https://github.com'` : git demande un
identifiant, et un notebook n'a pas d'entree interactive. Deux facons de s'en
sortir, la cellule de recuperation du code gere les deux :

1. **rendre le depot public** (Settings > General > Change visibility). Il ne
   contient que du code, aucune donnee ni aucun poids ;
2. **garder le depot prive et donner un jeton a Colab** : creer un jeton
   fine-grained sur github.com/settings/tokens avec l'acces `Contents: read` sur
   ce depot, puis le deposer dans les secrets Colab (icone cle, a gauche) sous le
   nom `GITHUB_TOKEN` en activant l'acces pour le notebook. Le jeton est injecte
   dans l'URL de clonage, jamais ecrit dans le notebook ni affiche.

**Les chemins Drive sont ecrits en clair dans la cellule `CONFIG DONNEES`** de
chaque notebook (memes valeurs que `comparaison_modeles_ultralytics.ipynb`) :
c'est le seul endroit a modifier pour changer de jeu de donnees. La cellule
suivante verifie que chaque chemin existe avant de lancer quoi que ce soit.
`aphids_det/config.py` porte les memes valeurs par defaut, pour un usage hors
notebook.

Avant une campagne, la plomberie se verifie sans GPU ni donnees :

```bash
python tests/test_pipeline.py   # folds, conversion COCO, metriques, CSV
python tests/test_figure.py     # mise en page de la figure d'augmentations
```

## Structure

```text
aphids_det/
  config.py        chemins, protocole, budget, seuils d'evaluation
  folds.py         folds depuis le CSV de split + listes de train figees
  cocoify.py       conversion YOLO -> COCO (deux arborescences)
  augment.py       reference d'augmentation + traductions par framework
  evaluate.py      evaluateur COCO unifie (mAP, P/R/F1, TP/FP/FN)
  visualize.py     figure comparative des augmentations des 4 pipelines
  bench.py         latence CPU, taille, CSV de resultats, boucle de CV
  external.py      clone des depots, poids COCO, relance sur OOM
  report.py        moyennes par modele, classeur Excel
  runners/         un module par framework (+ detr_repo.py partage)
  assets/          code injecte dans les depots externes
notebooks/         un notebook Colab par framework
tools/             regeneration des notebooks et de la doc
tests/             test hors-ligne de la chaine complete
```

## Voir les augmentations

`notebooks/07_visualisation_augmentations.ipynb` produit une figure ou chaque
ligne est un pipeline et chaque colonne un tirage aleatoire sur la meme tuile.
Chaque ligne execute le **vrai** code du framework : `YOLODataset` d'Ultralytics
(mosaique comprise), les transforms albumentations de `aug_config` pour RF-DETR,
les classes `torchvision.transforms.v2` de la liste d'ops des depots pour
RT-DETR/D-FINE, et `random_affine` + `augment_hsv` de YOLOX orchestres comme sa
`MosaicDetection`. Les quatre cohabitent dans un meme runtime parce qu'aucun
entrainement n'a lieu : YOLOX y est clone sans etre installe.

C'est le moyen le plus rapide de verifier de visu ce que dit
[docs/AUGMENTATION.md](docs/AUGMENTATION.md) : mosaique presente chez YOLO et
YOLOX, absente des trois DETR ; miroirs verticaux ajoutes partout ; recadrages
`ZoomOut`/`IoUCrop` a la place de `translate`/`scale` chez les DETR.

## Sorties (sur le Drive, dans `OUT_DIR`)

| Fichier | Contenu |
|---|---|
| `benchmark_detection_cv.csv` | une ligne par (modele, fold) : metriques, cout, provenance |
| `predictions_coco/<modele>_fold<f>.json` | detections brutes, re-evaluables a volonte |
| `modeles_detection/` | meilleurs poids par (modele, fold) |
| `splits_figes/` | listes de train partagees par tous les modeles |
| `augmentations_detection.csv` | table de correspondance des augmentations |
| `benchmark_detection_<date>.xlsx` | 4 feuilles : hyperparametres, augmentations, resultats par fold, moyennes |

Colonnes de cout du CSV : `train_time_s`, `latency_cpu_ms`, `latency_std_ms`,
`n_params_M`, `size_MB` (poids float32 du modele), `ckpt_MB` (fichier sauvegarde),
plus `batch`, `grad_accum`, `best_epoch`, `epochs_run` et `repo_commit`.

## Points a connaitre avant de publier les chiffres

- **Mosaique.** RF-DETR, RT-DETR et D-FINE n'en proposent aucune : les trois
  s'entrainent sans, les YOLO et YOLOX avec `mosaic = 1`. C'est le principal
  ecart d'augmentation qui subsiste.
- **Early stopping.** Natif chez Ultralytics et RF-DETR (patience 5). RT-DETR,
  D-FINE et YOLOX n'en ont pas : ils consomment les 30 epoques, et le benchmark
  retient l'epoque de meilleur mAP de validation. La selection du modele est
  donc la meme partout, le cout en calcul non.
- **Batch effectif.** 32 pour tout le monde. RF-DETR y arrive par accumulation
  (8 x 4) ; RT-DETR, D-FINE et YOLOX n'ont pas d'accumulation, leur batch
  physique est donc leur batch effectif. En cas d'OOM CUDA, le batch est divise
  par deux automatiquement (jusqu'a deux fois) et la valeur reellement utilisee
  est enregistree dans le CSV.
- **RT-DETR : variante v2 par defaut.** `rtdetr_pytorch` (le RT-DETR-R18 de
  l'article) importe `torchvision.datapoints`, supprime de torchvision depuis la
  0.17 : il ne demarre pas sur un Colab recent. Le benchmark utilise donc
  `rtdetrv2_pytorch` avec le meme backbone R18. Pour forcer la v1 dans un
  environnement torchvision <= 0.16 :
  `rtdetr_runner.run_cv(variant="rtdetr_v1")`.
- **YOLOX.** Depot non maintenu : il demande `numpy < 2`, donc un redemarrage du
  runtime Colab apres installation. Sa resolution par defaut (416) est portee a
  640 pour rester comparable, et son mixup est desactive.
- **Chiffres non comparables a ceux des anciens notebooks.** `comparaison_modeles_cv.csv`
  contenait des metriques natives (Ultralytics `.val()` d'un cote, COCOeval de
  l'autre). Le benchmark ecrit dans un fichier distinct,
  `benchmark_detection_cv.csv`.

## Regenerer notebooks et documentation

```bash
python tools/make_notebooks.py   # notebooks/ a partir du code
python tools/make_docs.py        # docs/AUGMENTATION.md a partir de augment.py
```

## Etat

Le protocole de donnees, la conversion COCO, l'evaluateur, le harnais de
resultats et la reecriture des transforms RT-DETR/D-FINE sont couverts par
`tests/test_pipeline.py`. La selection des tuiles et la mise en page de la
figure d'augmentations le sont par `tests/test_figure.py` (pipelines remplaces
par des imitations numpy, pour tester le rendu sans installer les frameworks).

Le reste — entrainements, inference, installation des depots, et les quatre
pipelines d'augmentation reels de `visualize.py` — a ete ecrit d'apres les
configs et les scripts officiels de chaque depot (relevees le 2026-09-20), mais
demande un environnement avec torch : ces parties n'ont pas encore ete executees.
Prevoir donc, dans cet ordre : `07_visualisation_augmentations.ipynb` (rapide,
sans GPU, et il valide les quatre pipelines d'augmentation), puis un seul fold
d'un modele (`run_fold(0)`), puis les 35 entrainements.
