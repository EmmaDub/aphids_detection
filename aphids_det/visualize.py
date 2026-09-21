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

Utilisation :

    from aphids_det import visualize
    visualize.compare(fold=0, n_aug=4, save=True)
"""

import random
import textwrap
from collections import namedtuple
from pathlib import Path

import numpy as np

from . import config as cfg, external
from .augment import (AUG, AUG_RFDETR, AUG_YOLOX, masque_visibilite,
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


PIPELINES = {
    "YOLO26n / YOLO11n / YOLO12n": aug_ultralytics,
    "RF-DETR-N": aug_rfdetr,
    "RT-DETR-R18 / D-FINE-N": aug_detr,
    "YOLOX-Nano": aug_yolox,
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
