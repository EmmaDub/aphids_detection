# -*- coding: utf-8 -*-
"""Visualisation comparative des augmentations des quatre pipelines du benchmark.

Chaque ligne de la figure est produite par le VRAI code d'augmentation du
framework concerne, pas par une imitation :

| Ligne | Code execute |
|---|---|
| YOLO26n / YOLO11n / YOLO12n | `ultralytics.data.YOLODataset` en mode `augment=True` avec `augment.AUG` (mosaique comprise) |
| RF-DETR-N | les transforms albumentations de `augment.AUG_RFDETR`, celles que recoit `model.train(aug_config=...)` |
| RT-DETR-R18 / D-FINE-N | les classes `torchvision.transforms.v2` de la liste d'ops des depots, reecrite par `detr_repo.patch_ops` |
| YOLOX-Nano | `random_affine` et `augment_hsv` du depot YOLOX, orchestres comme `MosaicDetection.__getitem__` |

Les seules approximations, et elles sont visibles sur la figure : le
redimensionnement interne de RF-DETR n'est pas reproduit, et les conversions de
fin de pipeline (tenseur, normalisation des boites) sont omises puisqu'elles
n'ont aucun effet visuel.

Deux figures :

    from aphids_det import visualize
    visualize.compare(fold=0, n_aug=4, save=True)   # tirages aleatoires
    visualize.compare_effets(fold=0, save=True)     # un effet par colonne, a sa
                                                    # valeur extreme, sans hasard
"""

import random
import textwrap
from contextlib import contextmanager
from collections import namedtuple
from pathlib import Path

import numpy as np

from . import config as cfg, external
from .augment import (AUG, AUG_RFDETR, AUG_YOLOX, MODELES,
                      masque_visibilite, patch_ultralytics_albumentations,
                      patch_ultralytics_visibility)
from .folds import fold_train_paths, has_annotations, label_of

Tile = namedtuple("Tile", "path boxes classes")

# --- Couleurs -------------------------------------------------------------
# Slots 1 et 2 du theme categoriel de reference (bleu, orange) : la paire est
# validee toutes-paires en clair comme en sombre. Le texte reste en encre, la
# couleur ne sert qu'a l'identite des classes, toujours doublee par la legende.
CLASS_COLORS = ("#2a78d6", "#eb6834")
INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#8a8880"
SURFACE = "#fcfcfb"

# --- Liste d'ops d'entrainement des depots RT-DETRv2 et D-FINE ------------
# Relevee le 2026-09-20 dans configs/rtdetrv2/include/dataloader.yml et
# configs/dfine/include/dataloader.yml : les deux depots ont la MEME liste.
# Si un depot est deja clone dans cfg.REPOS_DIR, sa version est relue a la
# place de cette copie (cf. _detr_ops).
DETR_BASE_OPS = [
    {"type": "RandomPhotometricDistort", "p": 0.5},
    {"type": "RandomZoomOut", "fill": 0},
    {"type": "RandomIoUCrop", "p": 0.8},
    {"type": "SanitizeBoundingBoxes", "min_size": 1},
    {"type": "RandomHorizontalFlip"},
    {"type": "Resize", "size": [640, 640]},
    {"type": "SanitizeBoundingBoxes", "min_size": 1},
    {"type": "ConvertPILImage", "dtype": "float32", "scale": True},
    {"type": "ConvertBoxes", "fmt": "cxcywh", "normalize": True},
]
# Conversions de fin de pipeline : aucun effet visuel, on ne les rejoue pas.
_SKIP_OPS = {"ConvertPILImage", "ConvertBoxes", "ConvertBox", "ToImageTensor",
             "ConvertDtype", "Normalize"}


# ========================================================================
# Tuiles sources
# ========================================================================
def _read_rgb(path):
    from PIL import Image
    return np.asarray(Image.open(path).convert("RGB"))


def _boxes_of(path):
    """(boites xyxy en pixels, classes) d'une tuile, depuis son label YOLO."""
    from PIL import Image

    with Image.open(path) as im:
        W, H = im.size
    boxes, classes = [], []
    lbl = label_of(path)
    if lbl.exists():
        for line in lbl.read_text().splitlines():
            s = line.split()
            if len(s) < 5:
                continue
            c = int(float(s[0]))
            xc, yc, bw, bh = (float(v) for v in s[1:5])
            boxes.append([(xc - bw / 2) * W, (yc - bh / 2) * H,
                          (xc + bw / 2) * W, (yc + bh / 2) * H])
            classes.append(c)
    return (np.array(boxes, dtype=np.float32).reshape(-1, 4),
            np.array(classes, dtype=np.int64))


def sample_tiles(fold=0, n=1, seed=0, min_boxes=2, pool_size=200):
    """Tuiles d'entrainement contenant des pucerons, plus un vivier pour la mosaique.

    Renvoie (tuiles choisies, vivier). Le vivier sert aux pipelines a mosaique
    (YOLO, YOLOX), qui composent chaque echantillon avec trois autres tuiles.
    """
    paths = [p for p in fold_train_paths(fold) if has_annotations(label_of(p))]
    if not paths:
        raise RuntimeError(
            f"Aucune tuile annotee dans le train du fold {fold}.\n"
            "Causes habituelles : folds non construits dans cette session "
            "(folds.build_folds()), ou labels non lies (colonne label_file du CSV "
            "de split / chemin labels_dir).\n"
            "Pour trancher : from aphids_det import folds ; folds.diagnose()")
    rng = random.Random(seed)
    rng.shuffle(paths)

    chosen, pool = [], []
    for p in paths:
        boxes, classes = _boxes_of(p)
        if len(boxes) == 0:
            continue
        tile = Tile(Path(p), boxes, classes)
        pool.append(tile)
        if len(chosen) < n and len(boxes) >= min_boxes:
            chosen.append(tile)
        if len(pool) >= pool_size and len(chosen) >= n:
            break
    if not chosen:                       # aucune tuile assez peuplee : on relache
        chosen = pool[:n]
    return chosen, pool


# ========================================================================
# Pipelines
# ========================================================================
def aug_ultralytics(tile, n_aug, pool, seed=0):
    """Vrai pipeline d'entrainement Ultralytics (mosaique + perspective + HSV + flips)."""
    from ultralytics.cfg import get_cfg
    from ultralytics.data.dataset import YOLODataset
    from ultralytics.utils import DEFAULT_CFG

    patch_ultralytics_visibility()      # meme seuil qu'a l'entrainement
    patch_ultralytics_albumentations()  # bloc cache neutralise, comme a l'entrainement
    # AUG est passe tel quel, en ignorant les cles que la version installee
    # ne connait pas (cutmix et bgr sont recents).
    over = {k: v for k, v in AUG.items() if hasattr(DEFAULT_CFG, k)}
    ignore = sorted(set(AUG) - set(over))
    hyp = get_cfg(DEFAULT_CFG, overrides={**over, "imgsz": cfg.IMGSZ})

    txt = Path(cfg.WORK_ROOT) / "viz" / "pool_ultralytics.txt"
    txt.parent.mkdir(parents=True, exist_ok=True)
    paths = [str(t.path) for t in pool]
    if str(tile.path) not in paths:
        paths.insert(0, str(tile.path))
    txt.write_text("\n".join(paths))

    names = {i: n for i, n in cfg.CLASS_NAMES.items()}
    ds = YOLODataset(img_path=str(txt), imgsz=cfg.IMGSZ, augment=True, hyp=hyp,
                     data={"names": names, "nc": len(names), "channels": 3},
                     task="detect", rect=False)
    idx = next((i for i, f in enumerate(ds.im_files)
                if Path(f).name == tile.path.name), 0)

    out = []
    for k in range(n_aug):
        random.seed(seed * 100 + k)      # tirages reproductibles
        np.random.seed(seed * 100 + k)
        d = ds[idx]
        img = d["img"].permute(1, 2, 0).numpy()          # RGB uint8
        H, W = img.shape[:2]
        b = d["bboxes"].numpy().reshape(-1, 4)           # xywh normalise
        xyxy = np.stack([(b[:, 0] - b[:, 2] / 2) * W, (b[:, 1] - b[:, 3] / 2) * H,
                         (b[:, 0] + b[:, 2] / 2) * W, (b[:, 1] + b[:, 3] / 2) * H],
                        axis=1) if len(b) else np.zeros((0, 4))
        out.append((np.ascontiguousarray(img), xyxy,
                    d["cls"].numpy().reshape(-1).astype(int)))
    if ignore:
        print(f"  Ultralytics : cles ignorees par cette version -> {', '.join(ignore)}")
    return out, "mosaique 4 tuiles + perspective + HSV + flips"


def aug_rfdetr(tile, n_aug, pool=None, seed=0):
    """Transforms albumentations passees a RF-DETR via `aug_config`."""
    import albumentations as A

    tfs = []
    for name, kw in AUG_RFDETR.items():
        klass = getattr(A, name, None)
        if klass is None:
            print(f"  transform albumentations inconnue, ignoree : {name}")
            continue
        tfs.append(klass(**kw))
    pipe = A.Compose(tfs, bbox_params=A.BboxParams(format="pascal_voc",
                                                   label_fields=["classes"]))
    img = _read_rgb(tile.path)
    out = []
    for k in range(n_aug):
        random.seed(seed * 100 + k)
        np.random.seed(seed * 100 + k)
        res = pipe(image=img, bboxes=tile.boxes.tolist(),
                   classes=tile.classes.tolist())
        out.append((np.ascontiguousarray(res["image"]),
                    np.array(res["bboxes"], dtype=np.float32).reshape(-1, 4),
                    np.array(res["classes"], dtype=int)))
    print("  RF-DETR : recadrage/resize interne du framework non reproduit ici")
    return out, "sans mosaique ; flips + HSV"


def _detr_ops():
    """Liste d'ops du depot si un clone est disponible, sinon la copie locale."""
    from .runners.detr_repo import SPECS, load_config_with_includes, patch_ops

    for variant in ("rtdetr_v2", "dfine_n"):
        spec = SPECS[variant]
        root = Path(cfg.REPOS_DIR) / external.REPOS[spec["repo"]]["dir"]
        work = root / spec["subdir"] if spec["subdir"] else root
        base = work / spec["base_cfg"]
        if base.exists():
            try:
                merged = load_config_with_includes(base)
                ops = merged["train_dataloader"]["dataset"]["transforms"]["ops"]
                return patch_ops(ops)[0], f"ops lues dans le depot clone ({variant})"
            except Exception as e:
                print(f"  lecture de {base} impossible ({e}) : copie locale utilisee")
    return patch_ops(DETR_BASE_OPS)[0], "ops de reference (copie du 2026-09-20)"


def _detr_transform(ops):
    """Compose torchvision v2 a partir de la liste d'ops du depot."""
    import inspect

    import torchvision.transforms.v2 as T

    tfs = []
    for op in ops:
        name = op.get("type")
        if name in _SKIP_OPS:
            continue
        kw = {k: v for k, v in op.items() if k != "type"}
        klass = getattr(T, name, None)
        if klass is None and name == "SanitizeBoundingBox":
            klass = T.SanitizeBoundingBoxes            # nom de RT-DETR v1
        if klass is None:
            print(f"  transform torchvision inconnue, ignoree : {name}")
            continue
        # Les depots ajoutent un `p` a RandomIoUCrop, que torchvision n'a pas :
        # on le rend par un RandomApply, ce que fait leur surcharge.
        p = kw.pop("p", None)
        if p is not None and "p" not in inspect.signature(klass.__init__).parameters:
            tfs.append(T.RandomApply([klass(**kw)], p=p))
        else:
            if p is not None:
                kw["p"] = p
            tfs.append(klass(**kw))
    return T.Compose(tfs)


def aug_detr(tile, n_aug, pool=None, seed=0):
    """Pipeline RT-DETR / D-FINE : les memes classes torchvision v2 que les depots."""
    import torch
    from torchvision import tv_tensors

    ops, source = _detr_ops()
    pipe = _detr_transform(ops)
    img = _read_rgb(tile.path)
    H, W = img.shape[:2]

    out = []
    for k in range(n_aug):
        torch.manual_seed(seed * 100 + k)
        random.seed(seed * 100 + k)
        sample = {
            "image": tv_tensors.Image(torch.from_numpy(img).permute(2, 0, 1)),
            "boxes": tv_tensors.BoundingBoxes(torch.from_numpy(tile.boxes),
                                              format="XYXY", canvas_size=(H, W)),
            "labels": torch.from_numpy(tile.classes),
        }
        res = pipe(sample)
        out.append((res["image"].permute(1, 2, 0).numpy().astype(np.uint8),
                    res["boxes"].numpy().reshape(-1, 4),
                    res["labels"].numpy().astype(int)))
    print(f"  RT-DETR / D-FINE : {source}, geometrie '{cfg.DETR_GEOM}'")
    geo = {"reference": "ZoomOut + IoUCrop cales sur la reference",
           "affine": "RandomAffine, parametres d'Ultralytics",
           "natif": "ZoomOut + IoUCrop aux bornes des depots"}
    return out, f"sans mosaique ; {geo.get(cfg.DETR_GEOM, cfg.DETR_GEOM)}"


def _yolox_funcs():
    """Fonctions d'augmentation du depot YOLOX (clone sans installation)."""
    repo = external.clone("yolox")
    external.add_to_path(repo)
    from yolox.data.data_augment import augment_hsv, random_affine
    from yolox.data.datasets.mosaicdetection import get_mosaic_coordinate
    return augment_hsv, random_affine, get_mosaic_coordinate


def aug_yolox(tile, n_aug, pool, seed=0):
    """Mosaique YOLOX : orchestration copiee de `MosaicDetection.__getitem__`.

    Puis `random_affine` et `augment_hsv` du depot, et les miroirs horizontal
    (natif) et vertical (ajoute par le benchmark), comme dans
    `assets/yolox_exp_aphids.py`.
    """
    import cv2

    augment_hsv, random_affine, get_mosaic_coordinate = _yolox_funcs()
    input_h = input_w = cfg.IMGSZ
    others = [t for t in pool if t.path != tile.path] or [tile]

    out = []
    for k in range(n_aug):
        rng = random.Random(seed * 100 + k)
        random.seed(seed * 100 + k)
        np.random.seed(seed * 100 + k)

        yc = int(rng.uniform(0.5 * input_h, 1.5 * input_h))
        xc = int(rng.uniform(0.5 * input_w, 1.5 * input_w))
        quatre = [tile] + [others[rng.randrange(len(others))] for _ in range(3)]

        mosaic_img, mosaic_labels = None, []
        for i_mosaic, t in enumerate(quatre):
            img = cv2.imread(str(t.path))                       # BGR, comme YOLOX
            h0, w0 = img.shape[:2]
            scale = min(input_h / h0, input_w / w0)
            img = cv2.resize(img, (int(w0 * scale), int(h0 * scale)),
                             interpolation=cv2.INTER_LINEAR)
            h, w, c = img.shape
            if i_mosaic == 0:
                mosaic_img = np.full((input_h * 2, input_w * 2, c), 114, dtype=np.uint8)
            (lx1, ly1, lx2, ly2), (sx1, sy1, sx2, sy2) = get_mosaic_coordinate(
                mosaic_img, i_mosaic, xc, yc, w, h, input_h, input_w)
            mosaic_img[ly1:ly2, lx1:lx2] = img[sy1:sy2, sx1:sx2]
            padw, padh = lx1 - sx1, ly1 - sy1
            lab = np.zeros((len(t.boxes), 5), dtype=np.float32)
            if len(t.boxes):
                lab[:, :4] = t.boxes * scale + np.array([padw, padh, padw, padh])
                lab[:, 4] = t.classes
            mosaic_labels.append(lab)

        mosaic_labels = np.concatenate(mosaic_labels, 0)
        np.clip(mosaic_labels[:, 0], 0, 2 * input_w, out=mosaic_labels[:, 0])
        np.clip(mosaic_labels[:, 1], 0, 2 * input_h, out=mosaic_labels[:, 1])
        np.clip(mosaic_labels[:, 2], 0, 2 * input_w, out=mosaic_labels[:, 2])
        np.clip(mosaic_labels[:, 3], 0, 2 * input_h, out=mosaic_labels[:, 3])

        aires = ((mosaic_labels[:, 2] - mosaic_labels[:, 0]) *
                 (mosaic_labels[:, 3] - mosaic_labels[:, 1])).copy()
        img_t, lab_t = random_affine(
            mosaic_img, mosaic_labels, target_size=(input_w, input_h),
            degrees=AUG_YOLOX["degrees"], translate=AUG_YOLOX["translate"],
            scales=tuple(AUG_YOLOX["mosaic_scale"]), shear=AUG_YOLOX["shear"])

        # Meme filtre que assets/yolox_exp_aphids.py : une boite qui ne conserve
        # pas MIN_VISIBILITY de son aire apres rognage n'est plus annotee.
        if len(lab_t) == len(aires):
            lab_t = lab_t[masque_visibilite(aires, lab_t[:, :4])]

        boxes = lab_t[:, :4].copy()
        classes = lab_t[:, 4].astype(int)
        if rng.random() < AUG_YOLOX["vflip_prob"]:             # ajout du benchmark
            img_t = img_t[::-1]
            boxes[:, 1::2] = img_t.shape[0] - boxes[:, 3::-2]
        img_t = np.ascontiguousarray(img_t)
        if rng.random() < AUG_YOLOX["hsv_prob"]:
            augment_hsv(img_t, *AUG_YOLOX["hsv_gains"])
        if rng.random() < AUG_YOLOX["flip_prob"]:              # _mirror de YOLOX
            img_t = img_t[:, ::-1]
            boxes[:, 0::2] = img_t.shape[1] - boxes[:, 2::-2]

        out.append((cv2.cvtColor(np.ascontiguousarray(img_t), cv2.COLOR_BGR2RGB),
                    boxes, classes))
    return out, "mosaique 4 tuiles + affine + HSV additif + flips"


# Les libelles viennent d'augment.MODELES : figures et table d'augmentation
# designent ainsi les modeles de la meme facon.
PIPELINES = {
    MODELES["ultralytics"]: aug_ultralytics,
    MODELES["rfdetr"]: aug_rfdetr,
    MODELES["detr"]: aug_detr,
    MODELES["yolox"]: aug_yolox,
}


# ========================================================================
# Figure
# ========================================================================
def _frame(ax):
    """Cadre discret, sans graduations : l'image porte l'information, pas les axes."""
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_edgecolor("#d8d7d2")
        spine.set_linewidth(0.6)


def _draw(ax, img, boxes, classes, lw=1.6):
    """Affiche une image et ses boites, halo blanc sous le trait pour la lisibilite."""
    import matplotlib.patheffects as pe
    from matplotlib.patches import Rectangle

    ax.imshow(img)
    _frame(ax)
    for (x1, y1, x2, y2), c in zip(np.asarray(boxes).reshape(-1, 4), classes):
        rect = Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, linewidth=lw,
                         edgecolor=CLASS_COLORS[int(c) % len(CLASS_COLORS)])
        rect.set_path_effects([pe.withStroke(linewidth=lw + 1.4, foreground="white"),
                               pe.Normal()])
        ax.add_patch(rect)


def compare(fold=0, n_aug=4, seed=0, pipelines=None, tile=None, pool=None,
            save=False, min_boxes=2, figsize_scale=2.4):
    """Figure comparative : une ligne par pipeline, une colonne par tirage.

    La premiere colonne montre la tuile d'origine, repetee sur chaque ligne pour
    que l'oeil compare chaque pipeline a sa source. Renvoie la figure.
    """
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    pipelines = pipelines or PIPELINES
    if tile is None or pool is None:
        tiles, vivier = sample_tiles(fold=fold, n=1, seed=seed, min_boxes=min_boxes)
        tile = tile or tiles[0]
        pool = pool or vivier
    original = _read_rgb(tile.path)

    resultats = {}
    for nom, fonction in pipelines.items():
        try:
            resultats[nom] = fonction(tile, n_aug, pool, seed=seed)
            print(f"  {nom} : OK")
        except Exception as e:
            resultats[nom] = (None, f"indisponible : {type(e).__name__}: {e}")
            print(f"  {nom} : ECHEC -> {type(e).__name__}: {e}")

    n_rows = len(resultats)
    n_cols = n_aug + 1
    fig, axes = plt.subplots(n_rows, n_cols, squeeze=False,
                             figsize=(figsize_scale * n_cols, figsize_scale * n_rows + 1.1))
    fig.patch.set_facecolor(SURFACE)

    for r, (nom, (echantillons, note)) in enumerate(resultats.items()):
        _draw(axes[r][0], original, tile.boxes, tile.classes)
        axes[r][0].set_ylabel(textwrap.fill(nom, 18), fontsize=9,
                              color=INK_PRIMARY, rotation=0, ha="right",
                              va="center", labelpad=10)
        axes[r][0].set_xlabel(textwrap.fill(note, 32), fontsize=7,
                              color=INK_MUTED, labelpad=4)
        for c in range(n_aug):
            ax = axes[r][c + 1]
            if echantillons is None:
                ax.set_facecolor(SURFACE)
                _frame(ax)
                if c == 0:
                    ax.text(0.5, 0.5, "pipeline indisponible", fontsize=8,
                            color=INK_MUTED, ha="center", va="center",
                            transform=ax.transAxes)
                continue
            img, boxes, classes = echantillons[c]
            _draw(ax, img, boxes, classes)
        if r == 0:
            axes[r][0].set_title("Tuile d'origine", fontsize=9, color=INK_SECONDARY)
            for c in range(n_aug):
                axes[r][c + 1].set_title(f"Tirage {c + 1}", fontsize=9,
                                         color=INK_SECONDARY)

    noms = [cfg.CLASS_NAMES[i] for i in sorted(cfg.CLASS_NAMES)]
    fig.legend(handles=[Line2D([0], [0], color=CLASS_COLORS[i], lw=2.2, label=n)
                        for i, n in enumerate(noms)],
               loc="lower center", ncol=len(noms), frameon=False, fontsize=9,
               labelcolor=INK_SECONDARY, bbox_to_anchor=(0.5, 0.005))
    fig.suptitle(f"Augmentations comparees - tuile {tile.path.name} (fold {fold})",
                 fontsize=11, color=INK_PRIMARY, y=0.998)
    fig.tight_layout(rect=(0.0, 0.035, 1.0, 0.985))

    if save:
        out = Path(save) if isinstance(save, (str, Path)) else \
            Path(cfg.OUT_DIR) / f"augmentations_fold{fold}_{tile.path.stem}.png"
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=200, facecolor=SURFACE, bbox_inches="tight")
        print("Figure ecrite ->", out)
    return fig


def dump_augmentations(pipeline, n=18, fold=0, seed=0, colonnes=6, save=True,
                       figsize_scale=2.2):
    """Grille de `n` tirages augmentes d'UN pipeline, boites dessinees.

    Verification n.1 exigee apres chaque modification de modele : on regarde ce
    que le modele va reellement voir, sur des tuiles reelles du jeu de donnees.
    `pipeline` est un libelle de `PIPELINES` (cf. augment.MODELES).
    """
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    fonction = PIPELINES[pipeline]
    tuiles, vivier = sample_tiles(fold=fold, n=1, seed=seed, min_boxes=1)
    tile = tuiles[0]
    echantillons, note = fonction(tile, n, vivier, seed=seed)

    lignes = (n + colonnes - 1) // colonnes
    fig, axes = plt.subplots(lignes, colonnes, squeeze=False,
                             figsize=(figsize_scale * colonnes,
                                      figsize_scale * lignes + 1.0))
    fig.patch.set_facecolor(SURFACE)
    for k in range(lignes * colonnes):
        ax = axes[k // colonnes][k % colonnes]
        if k < len(echantillons):
            _draw(ax, *echantillons[k][:3])
            ax.set_title(f"tirage {k + 1}", fontsize=7.5, color=INK_SECONDARY)
        else:
            ax.set_facecolor(SURFACE)
            _frame(ax)

    noms = [cfg.CLASS_NAMES[i] for i in sorted(cfg.CLASS_NAMES)]
    fig.legend(handles=[Line2D([0], [0], color=CLASS_COLORS[i], lw=2.2, label=nm)
                        for i, nm in enumerate(noms)],
               loc="lower center", ncol=len(noms), frameon=False, fontsize=9,
               labelcolor=INK_SECONDARY, bbox_to_anchor=(0.5, 0.005))
    fig.suptitle(f"{pipeline} - {n} tirages sur {tile.path.name} ({note})",
                 fontsize=10.5, color=INK_PRIMARY, y=0.998)
    fig.tight_layout(rect=(0.0, 0.04, 1.0, 0.98))

    if save:
        slug = "".join(c if c.isalnum() else "_" for c in pipeline).strip("_")
        out = Path(save) if isinstance(save, (str, Path)) else \
            Path(cfg.OUT_DIR) / f"verif_augmentation_{slug}_fold{fold}.png"
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=170, facecolor=SURFACE, bbox_inches="tight")
        print(f"  dump visuel ({n} tirages) -> {out}")
    return fig


# ========================================================================
# Comparaison effet par effet, a valeur extreme (sans aleatoire)
# ========================================================================
# `compare()` montre des tirages aleatoires : deux lignes ne sont donc jamais
# comparables case par case. Ici chaque colonne est UN effet pousse a sa borne
# de la reference (saturation x1.2, echelle x0.75 et x1.25, translation +10 %...)
# et chaque ligne l'applique avec le code du framework concerne. Une case vide
# signifie que le framework n'expose pas ce reglage.
#
# Le determinisme est obtenu de deux facons, jamais en trichant sur le calcul :
#   - en passant un intervalle degenere quand l'API l'accepte -- ColorJitter
#     (saturation=(1.2, 1.2)), RandomZoomOut (side_range=(1.333, 1.333)),
#     `get_aug_params` de YOLOX (scales=(1.25, 1.25)) ;
#   - en forcant les tirages a leur borne quand elle ne l'accepte pas, par le
#     gestionnaire `_tirages_extremes` (Ultralytics tire son gain HSV et son
#     echelle avec `random.uniform` / `np.random.uniform` sans parametre de
#     borne fixe).

# (titre de colonne, cle interne, nom de l'effet dans augment.table())
# Le troisieme champ garantit que la figure et la table d'augmentation parlent
# de la meme chose : tests/test_figure.py verifie la correspondance.
EFFETS = [
    ("Saturation x1.2", "saturation", "Saturation"),
    ("Luminosite / valeur x1.2", "luminosite", "Luminosite / valeur"),
    ("Miroir vertical", "flipud", "Miroir vertical"),
    ("Miroir horizontal", "fliplr", "Miroir horizontal"),
    ("Zoom / echelle x0.75", "echelle_min", "Zoom / echelle"),
    ("Zoom / echelle x1.25", "echelle_max", "Zoom / echelle"),
    ("Translation +10%", "translation", "Translation"),
    ("Mosaique", "mosaique", "Mosaique"),
]

_GAIN_SAT = 1 + AUG["hsv_s"]        # 1.2
_GAIN_VAL = 1 + AUG["hsv_v"]        # 1.2
_ECHELLE = {"echelle_min": 1 - AUG["scale"], "echelle_max": 1 + AUG["scale"]}


@contextmanager
def _tirages_extremes(borne="max", valeur_torch=0.5):
    """Force les tirages aleatoires a leur borne, le temps d'un appel.

    `borne` vaut "max" (borne superieure de chaque `uniform`), "min" (borne
    inferieure) ou "milieu" (moyenne des deux, utile pour centrer une mosaique).
    `torch.rand` est fixe a `valeur_torch` (0.5 = recadrage centre cote
    torchvision) et `np.random.randint` a 1, pour que le decalage HSV de YOLOX
    porte sur tous ses canaux.

    Exception sur l'intervalle (0, 1), toujours ramene a 1.0 : Ultralytics
    l'utilise comme tirage de probabilite, et notamment pour decider de la
    conversion BGR -> RGB (`random.uniform(0, 1) > hyp.bgr`, avec bgr = 0).
    Le forcer a 0 rendait des images aux canaux rouge et bleu echanges, et
    reveillait les transforms desactivees (p = 0).
    """
    import random as _random

    import numpy as _np

    ru, nu, nri = _random.uniform, _np.random.uniform, _np.random.randint
    try:
        import torch as _torch
        tr = _torch.rand
    except ImportError:
        _torch, tr = None, None

    def _borne(a, b):
        if borne == "max":
            return b
        if borne == "min":
            return a
        return (a + b) / 2

    def faux_uniform(a, b):
        if (float(a), float(b)) == (0.0, 1.0):     # tirage de probabilite
            return 1.0
        return _borne(a, b)

    def faux_np_uniform(low=0.0, high=1.0, size=None):
        v = _borne(low, high)
        return _np.full(size, v, dtype=float) if size is not None else float(v)

    def faux_randint(low, high=None, size=None, dtype=int):
        return _np.ones(size, dtype=int) if size is not None else 1

    def faux_rand(*taille, **kw):
        if len(taille) == 1 and isinstance(taille[0], (tuple, list)):
            taille = tuple(taille[0])
        kw = {k: v for k, v in kw.items() if k in ("dtype", "device")}
        return _torch.full(taille or (1,), float(valeur_torch), **kw)

    _random.uniform, _np.random.uniform, _np.random.randint = (
        faux_uniform, faux_np_uniform, faux_randint)
    if _torch is not None:
        _torch.rand = faux_rand
    try:
        yield
    finally:
        _random.uniform, _np.random.uniform, _np.random.randint = ru, nu, nri
        if _torch is not None:
            _torch.rand = tr


# ------------------------------------------------------------- Ultralytics
_DS_CACHE = {}


def _ultralytics_dataset(tile, pool, over):
    """Dataset Ultralytics minimal, avec toutes les augmentations a zero sauf `over`."""
    from ultralytics.cfg import get_cfg
    from ultralytics.data.dataset import YOLODataset
    from ultralytics.utils import DEFAULT_CFG

    cle = (str(tile.path), tuple(sorted(over.items())))
    if cle in _DS_CACHE:
        return _DS_CACHE[cle]

    patch_ultralytics_visibility()
    patch_ultralytics_albumentations()
    zero = {k: (0.0 if isinstance(getattr(DEFAULT_CFG, k), (int, float)) else
                getattr(DEFAULT_CFG, k))
            for k in AUG if hasattr(DEFAULT_CFG, k)}
    zero.update({k: v for k, v in over.items() if hasattr(DEFAULT_CFG, k)})
    hyp = get_cfg(DEFAULT_CFG, overrides={**zero, "imgsz": cfg.IMGSZ})

    txt = Path(cfg.WORK_ROOT) / "viz" / "pool_effets.txt"
    txt.parent.mkdir(parents=True, exist_ok=True)
    chemins = [str(tile.path)] + [str(t.path) for t in (pool or [])[:3]
                                  if t.path != tile.path]
    txt.write_text("\n".join(chemins))

    noms = dict(cfg.CLASS_NAMES)
    ds = YOLODataset(img_path=str(txt), imgsz=cfg.IMGSZ, augment=True, hyp=hyp,
                     data={"names": noms, "nc": len(noms), "channels": 3},
                     task="detect", rect=False)
    idx = next((i for i, f in enumerate(ds.im_files)
                if Path(f).name == tile.path.name), 0)
    _DS_CACHE[cle] = (ds, idx)
    return ds, idx


def _ultralytics_sortie(ds, idx):
    d = ds[idx]
    img = np.ascontiguousarray(d["img"].permute(1, 2, 0).numpy())
    H, W = img.shape[:2]
    b = d["bboxes"].numpy().reshape(-1, 4)
    xyxy = (np.stack([(b[:, 0] - b[:, 2] / 2) * W, (b[:, 1] - b[:, 3] / 2) * H,
                      (b[:, 0] + b[:, 2] / 2) * W, (b[:, 1] + b[:, 3] / 2) * H], 1)
            if len(b) else np.zeros((0, 4)))
    return img, xyxy, d["cls"].numpy().reshape(-1).astype(int)


def effet_ultralytics(tile, effet, pool=None):
    """Un effet isole, a sa borne, par le pipeline Ultralytics."""
    reglages = {
        "saturation": ({"hsv_s": AUG["hsv_s"]}, "max"),
        "luminosite": ({"hsv_v": AUG["hsv_v"]}, "max"),
        "flipud": ({"flipud": 1.0}, "max"),
        "fliplr": ({"fliplr": 1.0}, "max"),
        "echelle_min": ({"scale": AUG["scale"]}, "min"),
        "echelle_max": ({"scale": AUG["scale"]}, "max"),
        "translation": ({"translate": AUG["translate"]}, "max"),
        # mosaique : centre de collage au milieu du canevas 1280, echelle 1,
        # et recadrage central a 640 -- c'est ce que fait RandomPerspective.
        "mosaique": ({"mosaic": 1.0}, "milieu"),
    }
    if effet not in reglages:
        return f"effet inconnu : {effet}"
    over, borne = reglages[effet]
    ds, idx = _ultralytics_dataset(tile, pool, over)
    random.seed(cfg.SEED)                   # choix des 3 autres tuiles de mosaique
    with _tirages_extremes(borne):
        return _ultralytics_sortie(ds, idx)


# ----------------------------------------------------------------- RF-DETR
def _rfdetr_branche_recadrage(tile):
    """Branche "resize -> recadrage -> resize 640" du OneOf interne de RF-DETR.

    C'est la seule variation d'echelle et de position que le framework applique,
    et elle n'est pas reglable depuis `aug_config`. On la construit avec le
    constructeur du package (`_build_train_resize_config`) et son propre wrapper
    albumentations, pour montrer ce qu'elle fait plutot que de laisser une case
    vide. Renvoie une chaine explicative si l'API du package a change.
    """
    import torch
    from PIL import Image

    try:
        from rfdetr.datasets.coco import _build_train_resize_config
        from rfdetr.datasets.transforms import AlbumentationsWrapper
    except Exception as e:
        return ("echelle et translation non\nreglables : elles viennent du\n"
                f"recadrage interne du framework\n({type(e).__name__})")

    try:
        conf = _build_train_resize_config([cfg.IMGSZ], square=True, scale_jitter=True)
        entree = conf[0]
        branche = entree["OneOf"]["transforms"][1]        # option B : avec recadrage
        wrappers = AlbumentationsWrapper.from_config([branche])
    except Exception as e:
        return ("echelle et translation non\nreglables : elles viennent du\n"
                "recadrage interne du framework\n"
                f"(structure illisible : {type(e).__name__})")

    img = Image.fromarray(_read_rgb(tile.path))
    cible = {"boxes": torch.from_numpy(tile.boxes.copy()),
             "labels": torch.from_numpy(tile.classes.copy()),
             "size": torch.tensor([cfg.IMGSZ, cfg.IMGSZ]),
             "orig_size": torch.tensor([cfg.IMGSZ, cfg.IMGSZ])}
    for w in wrappers:
        img, cible = w(img, cible)
    boxes = np.asarray(cible["boxes"], dtype=np.float32).reshape(-1, 4)
    classes = np.asarray(cible["labels"], dtype=int).reshape(-1)
    return (np.asarray(img.convert("RGB")), boxes, classes,
            "a la place : branche recadrage\ndu OneOf interne (resize 400-600,\n"
            "recadrage, retour a 640)")


def effet_rfdetr(tile, effet, pool=None):
    """Un effet isole, a sa borne, avec les transforms albumentations de `aug_config`."""
    import albumentations as A

    if effet == "saturation":
        t = A.ColorJitter(brightness=(1, 1), contrast=(1, 1),
                          saturation=(_GAIN_SAT, _GAIN_SAT), hue=(0, 0), p=1.0)
    elif effet == "luminosite":
        t = A.RandomBrightnessContrast(brightness_limit=(AUG["hsv_v"], AUG["hsv_v"]),
                                       contrast_limit=(0, 0), p=1.0)
    elif effet == "flipud":
        t = A.VerticalFlip(p=1.0)
    elif effet == "fliplr":
        t = A.HorizontalFlip(p=1.0)
    elif effet == "mosaique":
        return "mosaique absente\ndu framework"
    else:
        # Echelle et translation ne sont pas reglables : ce que RF-DETR fait A
        # LA PLACE, c'est la branche B de son OneOf interne (redimensionnement
        # intermediaire -> recadrage -> retour a 640). On la joue avec le
        # constructeur de pipeline du package lui-meme.
        return _rfdetr_branche_recadrage(tile)

    pipe = A.Compose([t], bbox_params=A.BboxParams(
        format="pascal_voc", label_fields=["classes"],
        min_visibility=cfg.MIN_VISIBILITY))
    res = pipe(image=_read_rgb(tile.path), bboxes=tile.boxes.tolist(),
               classes=tile.classes.tolist())
    return (np.ascontiguousarray(res["image"]),
            np.array(res["bboxes"], dtype=np.float32).reshape(-1, 4),
            np.array(res["classes"], dtype=int))


# --------------------------------------------------------- RT-DETR / D-FINE
def effet_detr(tile, effet, pool=None):
    """Un effet isole, a sa borne, avec les classes torchvision v2 des depots."""
    resize = {"type": "Resize", "size": [cfg.IMGSZ, cfg.IMGSZ]}
    sanitize = {"type": "SanitizeBoundingBoxes", "min_size": 1}
    note = None
    if effet == "saturation":
        ops = [{"type": "ColorJitter", "saturation": [_GAIN_SAT, _GAIN_SAT]}]
    elif effet == "luminosite":
        ops = [{"type": "ColorJitter", "brightness": [_GAIN_VAL, _GAIN_VAL]}]
    elif effet == "flipud":
        ops = [{"type": "RandomVerticalFlip", "p": 1.0}]
    elif effet == "fliplr":
        ops = [{"type": "RandomHorizontalFlip", "p": 1.0}]
    elif effet == "echelle_min":
        # toile agrandie de 1/0.75 puis retour a 640 : objet a x0.75
        r = round(1 / _ECHELLE["echelle_min"], 3)
        ops = [{"type": "RandomZoomOut", "fill": 114, "side_range": [r, r], "p": 1.0},
               resize]
    elif effet == "echelle_max":
        # recadrage de 1/1.25 = 80 % du cote puis retour a 640 : objet a x1.25
        r = round(1 / _ECHELLE["echelle_max"], 3)
        ops = [{"type": "RandomIoUCrop", "min_scale": r, "max_scale": r,
                "min_aspect_ratio": 0.9, "max_aspect_ratio": 1.1,
                "sampler_options": [0.0], "p": 1.0}, sanitize, resize]
    elif effet == "mosaique":
        return "mosaique absente\ndes deux depots"
    elif effet == "translation":
        # Pas de reglage de translation : ce que le depot fait A LA PLACE, c'est
        # tirer la position de son recadrage. Recadrage de 90 % du cote (donc un
        # decalage maximal de 10 %), pousse a son offset maximal.
        r = round(1 - AUG["translate"], 3)
        ops = [{"type": "RandomIoUCrop", "min_scale": r, "max_scale": r,
                "min_aspect_ratio": 0.9, "max_aspect_ratio": 1.1,
                "sampler_options": [0.0], "p": 1.0}, sanitize, resize]
        note = "a la place : recadrage IoU au\ndecalage maxi (+10 %), avec un\nzoom x1.11 induit par le retour a 640"
    else:
        return f"effet inconnu : {effet}"

    import torch
    from torchvision import tv_tensors

    img = _read_rgb(tile.path)
    H, W = img.shape[:2]
    echantillon = {
        "image": tv_tensors.Image(torch.from_numpy(img).permute(2, 0, 1)),
        "boxes": tv_tensors.BoundingBoxes(torch.from_numpy(tile.boxes),
                                          format="XYXY", canvas_size=(H, W)),
        "labels": torch.from_numpy(tile.classes),
    }
    # position du recadrage : centree pour un effet isole, poussee au maximum
    # quand c'est justement le decalage qu'on veut montrer
    position = 1.0 if effet == "translation" else 0.5
    with _tirages_extremes("max", valeur_torch=position):
        res = _detr_transform(ops)(echantillon)
    sortie = (res["image"].permute(1, 2, 0).numpy().astype(np.uint8),
              res["boxes"].numpy().reshape(-1, 4),
              res["labels"].numpy().astype(int))
    return sortie + (note,) if note else sortie


# -------------------------------------------------------------------- YOLOX
def effet_yolox(tile, effet, pool=None):
    """Un effet isole, a sa borne, avec `augment_hsv` et `random_affine` de YOLOX."""
    import cv2

    augment_hsv, random_affine, _ = _yolox_funcs()
    img = cv2.imread(str(tile.path))                   # BGR, comme YOLOX
    boxes = tile.boxes.copy()
    classes = tile.classes.copy()

    if effet in ("saturation", "luminosite"):
        _, sgain, vgain = AUG_YOLOX["hsv_gains"]
        gains = (0, sgain, 0) if effet == "saturation" else (0, 0, vgain)
        img = np.ascontiguousarray(img)
        with _tirages_extremes("max"):                 # decalage maximal, tous canaux
            augment_hsv(img, *gains)
    elif effet == "flipud":
        img = np.ascontiguousarray(img[::-1])
        boxes[:, 1::2] = img.shape[0] - tile.boxes[:, 3::-2]
    elif effet == "fliplr":
        img = np.ascontiguousarray(img[:, ::-1])
        boxes[:, 0::2] = img.shape[1] - tile.boxes[:, 2::-2]
    elif effet in ("echelle_min", "echelle_max", "translation"):
        s = 1.0 if effet == "translation" else _ECHELLE[effet]
        # `random_affine` de YOLOX met l'echelle a l'origine (0, 0) : on recentre
        # par son propre parametre translate, (1 - s) / 2, force a sa borne.
        t = AUG["translate"] if effet == "translation" else (1 - s) / 2
        lab = np.hstack([boxes, classes[:, None]]).astype(np.float32)
        with _tirages_extremes("max"):
            img, lab = random_affine(img, lab, target_size=(cfg.IMGSZ, cfg.IMGSZ),
                                     degrees=0.0, translate=t, scales=(s, s),
                                     shear=0.0)
        boxes, classes = lab[:, :4], lab[:, 4].astype(int)
    elif effet == "mosaique":
        # Mosaique 2x2 a centre fixe (milieu du canevas 1280), puis l'affine de
        # YOLOX a l'echelle 1. Son affine etant ancree en (0, 0), un translate
        # de -0.5 ramene la fenetre 640 au centre du canevas -- ce que fait
        # RandomPerspective d'Ultralytics par construction : les deux cases
        # montrent alors la meme zone, et sont comparables.
        quatre = [tile] + [t for t in (pool or []) if t.path != tile.path][:3]
        while len(quatre) < 4:
            quatre.append(tile)
        canevas = np.full((2 * cfg.IMGSZ, 2 * cfg.IMGSZ, 3), 114, np.uint8)
        etiquettes = []
        for i, t in enumerate(quatre):
            im = cv2.imread(str(t.path))
            h, w = im.shape[:2]
            ox, oy = (i % 2) * cfg.IMGSZ, (i // 2) * cfg.IMGSZ
            canevas[oy:oy + h, ox:ox + w] = im
            lab = np.zeros((len(t.boxes), 5), np.float32)
            if len(t.boxes):
                lab[:, :4] = t.boxes + np.array([ox, oy, ox, oy])
                lab[:, 4] = t.classes
            etiquettes.append(lab)
        lab = np.concatenate(etiquettes, 0)
        with _tirages_extremes("max"):
            img, lab = random_affine(canevas, lab, target_size=(cfg.IMGSZ, cfg.IMGSZ),
                                     degrees=0.0, translate=-0.5, scales=(1.0, 1.0),
                                     shear=0.0)
        boxes, classes = lab[:, :4], lab[:, 4].astype(int)
    else:
        return f"effet inconnu : {effet}"

    return cv2.cvtColor(np.ascontiguousarray(img), cv2.COLOR_BGR2RGB), boxes, classes


EFFET_PIPELINES = {
    MODELES["ultralytics"]: effet_ultralytics,
    MODELES["rfdetr"]: effet_rfdetr,
    MODELES["detr"]: effet_detr,
    MODELES["yolox"]: effet_yolox,
}


def compare_effets(fold=0, tile=None, pool=None, effets=None, pipelines=None,
                   seed=0, save=False, min_boxes=2, figsize_scale=2.2):
    """Figure deterministe : une ligne par pipeline, une colonne par effet extreme.

    Contrairement a `compare()`, aucune case n'est aleatoire : chaque colonne
    montre le meme effet pousse a la meme borne de la reference, ce qui rend les
    lignes comparables case par case. Une case barree signale un reglage que le
    framework n'expose pas.
    """
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    effets = effets or EFFETS
    pipelines = pipelines or EFFET_PIPELINES
    if tile is None or pool is None:
        tuiles, vivier = sample_tiles(fold=fold, n=1, seed=seed, min_boxes=min_boxes)
        tile = tile or tuiles[0]
        pool = pool or vivier
    original = _read_rgb(tile.path)

    resultats = {}
    for nom, fonction in pipelines.items():
        ligne = {}
        for entree in effets:
            effet = entree[1]
            try:
                ligne[effet] = fonction(tile, effet, pool)
            except Exception as e:
                print(f"  {nom} / {effet} : {type(e).__name__}: {e}")
                ligne[effet] = f"echec : {type(e).__name__}"
        faits = sum(1 for v in ligne.values() if not isinstance(v, (str, type(None))))
        print(f"  {nom} : {faits}/{len(effets)} effets reproduits")
        resultats[nom] = ligne

    n_rows, n_cols = len(resultats), len(effets) + 1
    fig, axes = plt.subplots(n_rows, n_cols, squeeze=False,
                             figsize=(figsize_scale * n_cols,
                                      figsize_scale * n_rows + 1.2))
    fig.patch.set_facecolor(SURFACE)

    for r, (nom, ligne) in enumerate(resultats.items()):
        _draw(axes[r][0], original, tile.boxes, tile.classes)
        axes[r][0].set_ylabel(textwrap.fill(nom, 18), fontsize=9,
                              color=INK_PRIMARY, rotation=0, ha="right",
                              va="center", labelpad=10)
        for c, entree in enumerate(effets, start=1):
            titre, effet = entree[0], entree[1]
            ax = axes[r][c]
            res = ligne.get(effet)
            if res is None or isinstance(res, str):
                # Pas d'image : le framework n'expose pas ce reglage et n'a rien
                # a montrer a la place. La raison est ecrite dans la case.
                ax.set_facecolor(SURFACE)
                _frame(ax)
                ax.text(0.5, 0.5, res or "reglage absent", fontsize=7,
                        color=INK_MUTED, ha="center", va="center",
                        transform=ax.transAxes, linespacing=1.5)
            else:
                # 4 elements = substitut : le framework n'a pas ce reglage, mais
                # voici ce qu'il fait a la place. La legende sous la case le dit.
                _draw(ax, *res[:3])
                if len(res) > 3 and res[3]:
                    ax.set_xlabel(res[3], fontsize=6.5, color=INK_MUTED,
                                  labelpad=3, linespacing=1.4)
            if r == 0:
                ax.set_title(titre, fontsize=8.5, color=INK_SECONDARY)
        if r == 0:
            axes[r][0].set_title("Tuile d'origine", fontsize=8.5, color=INK_SECONDARY)

    noms = [cfg.CLASS_NAMES[i] for i in sorted(cfg.CLASS_NAMES)]
    fig.legend(handles=[Line2D([0], [0], color=CLASS_COLORS[i], lw=2.2, label=n)
                        for i, n in enumerate(noms)],
               loc="lower center", ncol=len(noms), frameon=False, fontsize=9,
               labelcolor=INK_SECONDARY, bbox_to_anchor=(0.5, 0.005))
    fig.suptitle("Chaque effet a sa valeur extreme de reference - "
                 f"tuile {tile.path.name} (fold {fold})",
                 fontsize=11, color=INK_PRIMARY, y=0.998)
    fig.tight_layout(rect=(0.0, 0.04, 1.0, 0.985))

    if save:
        out = Path(save) if isinstance(save, (str, Path)) else \
            Path(cfg.OUT_DIR) / f"effets_fold{fold}_{tile.path.stem}.png"
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=200, facecolor=SURFACE, bbox_inches="tight")
        print("Figure ecrite ->", out)
    return fig
