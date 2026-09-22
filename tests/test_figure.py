"""Test de la mise en page de visualize.compare avec des pipelines factices.

    python tests/test_figure.py

Ne necessite ni torch ni albumentations : on remplace les quatre pipelines par
des transformations numpy qui imitent leurs effets (mosaique, flips, recadrage),
pour verifier le rendu de la figure (libelles, notes, legende, halo des boites,
cas "pipeline indisponible") et la selection des tuiles.
"""
import random
import sys
import tempfile
from pathlib import Path

import matplotlib
import numpy as np
from PIL import Image, ImageDraw

matplotlib.use("Agg")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import aphids_det.config as cfg

tmp = Path(tempfile.mkdtemp())
cfg.WORK_ROOT = tmp / "work"
cfg.OUT_DIR = tmp / "out"
cfg.refresh()

from aphids_det import visualize  # noqa: E402

# ---------------------------------------------------- tuiles synthetiques
N_TILES, S = 8, cfg.IMGSZ
root = Path(cfg.FOLDS_ROOT) / "fold_0"
(root / "images").mkdir(parents=True, exist_ok=True)
(root / "labels").mkdir(parents=True, exist_ok=True)

rng = random.Random(0)
for i in range(N_TILES):
    x = np.linspace(0, 1, S)[None, :]
    y = np.linspace(0, 1, S)[:, None]
    bg = np.stack([60 + 50 * x + 20 * y + 0 * y, 110 + 60 * y + 0 * x,
                   55 + 40 * x + 0 * y], -1)
    bg = np.broadcast_to(bg, (S, S, 3)).copy()
    bg += np.random.RandomState(i).normal(0, 5, bg.shape)
    img = Image.fromarray(np.clip(bg, 0, 255).astype("uint8"))
    d = ImageDraw.Draw(img)
    for _ in range(6):
        d.line([(rng.randint(0, S), 0), (rng.randint(0, S), S)],
               fill=(90, 150, 80), width=3)
    lignes = []
    for _ in range(rng.randint(2, 4)):
        cls = rng.choice([0, 0, 1])
        w = rng.randint(30, 60)
        h = int(w * rng.uniform(0.6, 1.0))
        cx, cy = rng.randint(w, S - w), rng.randint(h, S - h)
        d.ellipse([cx - w // 2, cy - h // 2, cx + w // 2, cy + h // 2],
                  fill=(150, 90, 60) if cls == 0 else (60, 60, 130),
                  outline=(30, 30, 30))
        lignes.append(f"{cls} {cx/S:.6f} {cy/S:.6f} {w/S:.6f} {h/S:.6f}")
    img.save(root / "images" / f"tuile_{i:02d}.jpg", quality=92)
    (root / "labels" / f"tuile_{i:02d}.txt").write_text("\n".join(lignes))

Path(cfg.SPLITS_FROZEN).mkdir(parents=True, exist_ok=True)
(Path(cfg.SPLITS_FROZEN) / f"train_fold0_neg{cfg.NEG_RATIO}.txt").write_text(
    "\n".join(str(root / "images" / f"tuile_{i:02d}.jpg") for i in range(N_TILES)))

# ---------------------------------------------------- selection des tuiles
tuiles, vivier = visualize.sample_tiles(fold=0, n=2, seed=0, min_boxes=2)
assert len(tuiles) == 2 and len(vivier) == N_TILES, (len(tuiles), len(vivier))
assert all(len(t.boxes) >= 2 for t in tuiles), "min_boxes non respecte"
assert all(len(t.boxes) == len(t.classes) for t in vivier)
t0 = tuiles[0]
print(f"sample_tiles OK : {t0.path.name}, {len(t0.boxes)} boites, "
      f"classes {sorted(set(t0.classes.tolist()))}, vivier {len(vivier)}")
b = t0.boxes
assert b[:, 0].min() >= 0 and b[:, 2].max() <= S, "boites hors tuile"

# ------------------------------------------- pipelines factices (sans torch)
def _flip_h(img, boxes):
    w = img.shape[1]
    nb = boxes.copy()
    nb[:, 0::2] = w - boxes[:, 2::-2]
    return img[:, ::-1].copy(), nb


def _flip_v(img, boxes):
    h = img.shape[0]
    nb = boxes.copy()
    nb[:, 1::2] = h - boxes[:, 3::-2]
    return img[::-1].copy(), nb


def faux_mosaique(tile, n_aug, pool, seed=0):
    """Imite une mosaique 2x2 puis un zoom central (effet YOLO / YOLOX)."""
    out = []
    r = random.Random(seed)
    for _ in range(n_aug):
        quatre = [tile] + [pool[r.randrange(len(pool))] for _ in range(3)]
        demi = S // 2
        toile = np.zeros((S, S, 3), np.uint8)
        boxes, classes = [], []
        for i, t in enumerate(quatre):
            im = np.asarray(Image.open(t.path).convert("RGB").resize((demi, demi)))
            ox, oy = (i % 2) * demi, (i // 2) * demi
            toile[oy:oy + demi, ox:ox + demi] = im
            for (x1, y1, x2, y2), c in zip(t.boxes, t.classes):
                boxes.append([x1 / 2 + ox, y1 / 2 + oy, x2 / 2 + ox, y2 / 2 + oy])
                classes.append(c)
        img, bx = toile, np.array(boxes, np.float32)
        if r.random() < 0.5:
            img, bx = _flip_h(img, bx)
        if r.random() < 0.5:
            img, bx = _flip_v(img, bx)
        out.append((img, bx, np.array(classes, int)))
    return out, "mosaique 4 tuiles + perspective + HSV + flips"


def faux_flips(tile, n_aug, pool=None, seed=0):
    """Imite un pipeline sans mosaique : flips + variation de luminosite."""
    out = []
    r = random.Random(seed + 7)
    base = visualize._read_rgb(tile.path)
    for _ in range(n_aug):
        img, bx = base.copy(), tile.boxes.copy()
        if r.random() < 0.5:
            img, bx = _flip_h(img, bx)
        if r.random() < 0.5:
            img, bx = _flip_v(img, bx)
        gain = r.uniform(0.8, 1.2)
        img = np.clip(img.astype(np.float32) * gain, 0, 255).astype(np.uint8)
        out.append((img, bx, tile.classes.copy()))
    return out, "sans mosaique ; flips + HSV"


def faux_crop(tile, n_aug, pool=None, seed=0):
    """Imite ZoomOut + IoUCrop : recadrage aleatoire puis retour a 640."""
    out = []
    r = random.Random(seed + 13)
    base = visualize._read_rgb(tile.path)
    for _ in range(n_aug):
        f = r.uniform(0.55, 0.95)
        cw = int(S * f)
        x0 = r.randint(0, S - cw)
        y0 = r.randint(0, S - cw)
        crop = base[y0:y0 + cw, x0:x0 + cw]
        img = np.asarray(Image.fromarray(crop).resize((S, S)))
        k = S / cw
        bx = (tile.boxes - np.array([x0, y0, x0, y0], np.float32)) * k
        garde = (bx[:, 2] > 4) & (bx[:, 3] > 4) & (bx[:, 0] < S - 4) & (bx[:, 1] < S - 4)
        out.append((img, np.clip(bx[garde], 0, S), tile.classes[garde]))
    return out, "sans mosaique ; ZoomOut + IoUCrop"


def pipeline_casse(tile, n_aug, pool=None, seed=0):
    raise RuntimeError("framework absent de ce runtime")


faux = {
    "YOLO26n / YOLO11n / YOLO12n": faux_mosaique,
    "RF-DETR-N": faux_flips,
    "RT-DETR-R18 / D-FINE-N": faux_crop,
    "YOLOX-Nano": faux_mosaique,
    "Pipeline en echec (test)": pipeline_casse,
}

def faux_effet(sans=(), substitut=()):
    """Pipeline d'effets factice.

    `sans` : effets sans image, la case affiche la raison.
    `substitut` : effets rendus par une alternative, la case porte une legende
    (4e element du tuple) -- c'est le cas "voici ce que le framework fait a la
    place".
    """
    def pipeline(tile, effet, pool=None):
        if effet in sans:
            return "reglage absent\ndu framework (test)"
        if effet in substitut:
            img = visualize._read_rgb(tile.path)
            return (img, tile.boxes.copy(), tile.classes.copy(),
                    "a la place : alternative\nnative du framework (test)")
        if effet == "mosaique":
            img = np.asarray(Image.open(tile.path).convert("RGB"))
            return img, tile.boxes.copy(), tile.classes.copy()
        img = visualize._read_rgb(tile.path)
        bx = tile.boxes.copy()
        if effet == "saturation":
            img = np.clip(img.astype(np.float32) * [1.2, 1.0, 1.0], 0, 255).astype(np.uint8)
        elif effet == "luminosite":
            img = np.clip(img.astype(np.float32) * 1.2, 0, 255).astype(np.uint8)
        elif effet == "flipud":
            img, bx = _flip_v(img, bx)
        elif effet == "fliplr":
            img, bx = _flip_h(img, bx)
        elif effet in ("echelle_min", "echelle_max", "translation"):
            k = {"echelle_min": 0.75, "echelle_max": 1.25, "translation": 1.0}[effet]
            d = 64 if effet == "translation" else 0
            m = np.full_like(img, 114)
            n = int(S * k)
            red = np.asarray(Image.fromarray(img).resize((n, n)))
            o = (S - n) // 2 + d
            xs, ys = max(o, 0), max(o, 0)
            xe, ye = min(o + n, S), min(o + n, S)
            m[ys:ye, xs:xe] = red[max(0, -o):ye - o, max(0, -o):xe - o]
            img = m
            bx = bx * k + (o - 0 if k != 1 else d)
        return img, bx, tile.classes.copy()
    return pipeline


png_effets = Path(cfg.OUT_DIR) / "figure_effets.png"
fig_effets = visualize.compare_effets(
    fold=0, tile=t0, pool=vivier, save=png_effets,
    pipelines={
        "YOLO26n / YOLO11n / YOLO12n": faux_effet(),
        "RF-DETR-N": faux_effet(sans=("mosaique",),
                                substitut=("echelle_min", "echelle_max",
                                           "translation")),
        "RT-DETR-R18 / D-FINE-N": faux_effet(sans=("mosaique",),
                                             substitut=("translation",)),
        "YOLOX-Nano": faux_effet(),
    })
attendu = len(visualize.EFFETS) + 1
assert fig_effets.axes[0].get_title() == "Tuile d'origine"

# les colonnes doivent porter le vocabulaire de la table d'augmentation
from aphids_det import augment                                    # noqa: E402
effets_table = set(augment.table()["effet"])
for titre, cle, nom_table in visualize.EFFETS:
    assert nom_table in effets_table, (nom_table, sorted(effets_table))
    assert titre.startswith(nom_table), (titre, nom_table)
print(f"colonnes conformes a augment.table() : "
      f"{[e[2] for e in visualize.EFFETS]}")

# les modeles portent le meme nom dans les figures et dans la table
assert list(visualize.PIPELINES) == list(visualize.EFFET_PIPELINES)        == list(augment.MODELES.values())
assert set(augment.MODELES.values()) <= set(augment.table().columns)
print(f"modeles nommes pareil partout : {list(augment.MODELES.values())}")
assert png_effets.exists() and png_effets.stat().st_size > 50_000
print(f"figure effets : {png_effets.name} "
      f"({png_effets.stat().st_size/1e3:.0f} ko, {attendu} colonnes)")
print("CHEMIN_PNG_EFFETS:", png_effets)

png = Path(cfg.OUT_DIR) / "figure_layout.png"
fig = visualize.compare(fold=0, n_aug=4, seed=0, pipelines=faux, tile=t0,
                        pool=vivier, save=png)
assert png.exists() and png.stat().st_size > 50_000, png.stat().st_size if png.exists() else 0
print(f"\nfigure : {png}  ({png.stat().st_size/1e3:.0f} ko, "
      f"{fig.get_size_inches()[0]:.1f}x{fig.get_size_inches()[1]:.1f} pouces)")
print("CHEMIN_PNG:", png)
