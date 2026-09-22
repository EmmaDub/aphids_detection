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

Ouvrir directement dans Colab, sans rien telecharger ni modifier :

| Notebook | Contenu | |
|---|---|---|
| 00 | folds, verifications, table d'augmentation -- **a lancer en premier** | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/EmmaDub/aphids_detection/blob/main/notebooks/00_preparation_folds.ipynb) |
| 01 | YOLO26n + YOLO11n + YOLO12n | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/EmmaDub/aphids_detection/blob/main/notebooks/01_ultralytics.ipynb) |
| 02 | RF-DETR Nano | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/EmmaDub/aphids_detection/blob/main/notebooks/02_rfdetr.ipynb) |
| 03 | RT-DETR-R18 | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/EmmaDub/aphids_detection/blob/main/notebooks/03_rtdetr_r18.ipynb) |
| 04 | D-FINE-N | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/EmmaDub/aphids_detection/blob/main/notebooks/04_dfine_n.ipynb) |
| 05 | YOLOX-Nano | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/EmmaDub/aphids_detection/blob/main/notebooks/05_yolox_nano.ipynb) |
| 06 | moyennes par modele + classeur Excel | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/EmmaDub/aphids_detection/blob/main/notebooks/06_synthese.ipynb) |
| 07 | voir les augmentations (sans GPU, independant) | [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/EmmaDub/aphids_detection/blob/main/notebooks/07_visualisation_augmentations.ipynb) |

Ces liens ouvrent **la derniere version poussee** : inutile de telecharger le
`.ipynb` puis de le modifier a la main, et c'est le moyen le plus sur d'eviter
d'invalider le fichier en l'editant hors de Colab. Pour repartir d'une version
propre apres avoir modifie un notebook : rouvrir le lien ci-dessus (Colab
propose alors une copie neuve), ou `git checkout notebooks/<fichier>.ipynb` en
local.

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
python tests/test_rebase.py     # relecture des listes figees d'une autre session
python tests/test_figure.py     # mise en page de la figure d'augmentations
python tests/test_notebooks.py  # notebooks : JSON, schema, cellules compilables
```

Un `.ipynb` edite a la main dans un editeur de texte se casse vite -- un
guillemet non echappe dans une ligne de code suffit a rendre le fichier illisible
par Colab, qui annonce un "notebook invalide". Les notebooks sont donc generes
par `tools/make_notebooks.py` : pour les modifier, editer le generateur et le
relancer, ou editer dans Colab (qui echappe le JSON correctement).
`tests/test_notebooks.py` verifie les deux.

**Si un fold parait vide**, `folds.diagnose(fold=0)` affiche en une cellule les
images et labels presents par fold, et ce que donnent les listes figees du Drive
une fois regreffees sur la racine de folds courante. Les listes figees
contiennent en effet des chemins absolus ecrits par la session qui les a creees
(les notebooks d'origine travaillaient dans `/content/kfold_yolo/...`, ce depot
dans `/content/aphids_work/folds/...`) : seule la queue du chemin
`fold_i/images/nom.jpg` est utilisee, et elle est regreffee automatiquement, de
sorte que la selection de tuiles figee est respectee a l'identique.

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

Le notebook produit deux figures complementaires :

- `visualize.compare()` -- des **tirages aleatoires** du pipeline complet : ce
  que le modele voit reellement a l'entrainement (mosaique comprise). Deux
  lignes ne sont pas comparables case par case.
- `visualize.compare_effets()` -- **un effet par colonne, pousse a la borne de
  la reference** (saturation x1.2, echelle x0.75 et x1.25, translation +10 %...)
  et sans aucun hasard : les lignes deviennent comparables case par case. C'est
  la figure qui verifie qu'un meme reglage produit bien le meme effet partout.
  Le determinisme vient d'un intervalle degenere quand l'API l'accepte
  (`saturation=(1.2, 1.2)`, `scales=(1.25, 1.25)`), et sinon des tirages forces
  a leur borne. Quand un framework n'expose pas le reglage, la case montre ce
  qu'il fait **a la place** (le recadrage `RandomIoUCrop` pousse a son decalage
  maximal pour RT-DETR et D-FINE, la branche recadrage du `OneOf` interne pour
  RF-DETR), avec une legende qui le dit ; elle ne reste vide que lorsqu'il n'y
  a aucune alternative, comme la mosaique.

Ensemble, elles verifient de visu ce que dit
[docs/AUGMENTATION.md](docs/AUGMENTATION.md) : mosaique presente chez YOLO et
YOLOX, absente des trois DETR ; miroirs verticaux ajoutes partout ; echelle
ramenee a x[0.75, 1.25] chez les DETR ; HSV additif de YOLOX contre gain
multiplicatif ailleurs.

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
- **Echelle des DETR.** Par defaut `RandomZoomOut` (toile jusqu'a x4) et
  `RandomIoUCrop` (recadrage jusqu'a 30 % du cote) font varier l'echelle dans
  x[0.25, 3.3], contre x[0.75, 1.25] pour la reference. `cfg.DETR_GEOM` regle ce
  point : `"reference"` (defaut) recale les bornes de ces deux transforms sur
  `scale` et `translate`, `"affine"` les remplace par un `RandomAffine` aux
  parametres exacts d'Ultralytics, `"natif"` restaure les amplitudes des depots.
- **Boites tronquees.** `cfg.MIN_VISIBILITY = 0.20` : une boite qui ne conserve
  pas 20 % de son aire apres augmentation n'est plus annotee. Le seuil est
  obtenu par quatre mecanismes differents, et un seul n'est pas exactement la
  meme regle :

  | Modele | Comment | Regle identique ? |
  |---|---|---|
  | YOLO26n/11n/12n | `box_candidates(area_thr=0.20)` -- le 0.10 code en dur est patche | oui |
  | RF-DETR-N | `min_visibility=0.20` impose au `BboxParams` du package (defaut 0.0) | oui |
  | YOLOX-Nano | filtre ajoute a la sortie de `random_affine` (ne retirait que les boites < 1 px) | oui |
  | RT-DETR-R18 / D-FINE-N | critere natif de `RandomIoUCrop` : seules les boites dont le **centre** tombe dans le recadrage sont gardees | **non, plus strict** |

  Pour les deux derniers, une boite conservee garde forcement au moins 25 % de
  son aire, donc le seuil n'est jamais viole ; mais une boite visible a plus de
  20 % dont le centre sort du recadrage est supprimee, la ou les autres
  modeles la garderaient. Ces deux modeles sont donc plus severes, jamais plus
  laxistes.
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
