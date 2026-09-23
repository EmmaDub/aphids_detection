# -*- coding: utf-8 -*-
"""Construction des folds a partir du CSV de split, et listes de train figees.

Portage direct de la logique des notebooks d'origine :
  - chaque ligne du CSV porte son fold (colonne `split`) ;
  - `label_file` vide  -> tuile de FOND (image seule, pas de label) ;
  - `label_file` rempli -> tuile PUCERON (image + label) ;
  - le renfort "cell" (`extra_train`) est ajoute au TRAIN uniquement.

La liste de train de chaque fold est ecrite une fois sur le Drive
(`SPLITS_FROZEN/train_fold{f}_neg{n}.txt`) puis relue telle quelle : tous les
modeles du benchmark s'entrainent donc exactement sur les memes tuiles.
"""

import random
import re
import zipfile
from pathlib import Path

import pandas as pd

from . import config as cfg

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


# ---------------------------------------------------------------- utilitaires
def has_annotations(lbl):
    """True si le fichier de label existe et contient au moins une boite."""
    lbl = Path(lbl)
    return lbl.exists() and any(l.strip() for l in lbl.read_text().splitlines())


def label_of(img_path):
    """Chemin du label YOLO d'une image (`<dossier>/images/x.jpg` -> `<dossier>/labels/x.txt`).

    Construit a partir des composants du chemin, et non par substitution de
    texte : le code doit marcher quel que soit le separateur de la plateforme.
    """
    p = Path(img_path)
    return p.parent.parent / "labels" / (p.stem + ".txt")


def _normalize_fold(v):
    """'fold3' -> 3 (tolere aussi un entier)."""
    return int(str(v).strip().lower().replace("fold", ""))


def _link(src, dst):
    """Lien symbolique src -> dst (ignore si dst existe deja)."""
    dst = Path(dst)
    if not dst.exists():
        try:
            dst.symlink_to(src)
        except (FileExistsError, OSError):
            pass


def _stage_dir(src):
    """Dossier LOCAL contenant les images de `src`.

    - si le dossier Drive est peuple -> on l'utilise directement ;
    - sinon si une archive zip existe -> extraction locale (une seule fois).
    """
    src = Path(src)
    if src.exists() and any(True for _ in src.iterdir()):
        return src
    local = Path(cfg.STAGE_LOCAL) / src.name
    if (local / ".done").exists():
        return local
    zip_src = Path(cfg.BG_ZIP)
    if zip_src.exists():
        local.mkdir(parents=True, exist_ok=True)
        print(f"  extraction de {zip_src} -> {local}")
        with zipfile.ZipFile(zip_src) as zf:
            zf.extractall(local)
        (local / ".done").write_text("ok")
        return local
    raise FileNotFoundError(f"Ni dossier peuple ni archive zip pour {src}")


def build_image_index(dirs):
    """Index {nom_de_fichier -> chemin} en fusionnant plusieurs dossiers sources."""
    idx = {}
    for d in dirs:
        base = _stage_dir(d)
        for p in Path(base).rglob("*"):
            if p.is_file() and p.suffix.lower() in IMG_EXTS:
                idx.setdefault(p.name, p)      # premier trouve gagne
    return idx


# ------------------------------------------------------------ folds depuis CSV
def build_folds(verbose=True):
    """Construit FOLDS_ROOT/fold_{0..5}/{images,labels} + _extra_train par symlinks."""
    variant = cfg.VARIANTS[cfg.SEARCH_VARIANT]
    root = Path(cfg.FOLDS_ROOT)
    df = pd.read_csv(variant["csv"])
    labels_dir = Path(variant["labels_dir"])

    if verbose:
        print("Indexation des images sources (pucerons + fonds)...")
    img_idx = build_image_index([cfg.MAIN_IMAGES_DIR, cfg.BG_DIR])
    if verbose:
        print(f"  {len(img_idx)} images indexees.")

    for f in range(cfg.N_CV_FOLDS + 1):        # folds 0..5 (5 = test)
        (root / f"fold_{f}" / "images").mkdir(parents=True, exist_ok=True)
        (root / f"fold_{f}" / "labels").mkdir(parents=True, exist_ok=True)

    ok = miss = 0
    for name, lf, split in zip(df["image"], df["label_file"], df["split"]):
        fold = _normalize_fold(split)
        src_img = img_idx.get(name)
        if src_img is None:
            miss += 1
            continue
        _link(src_img, root / f"fold_{fold}" / "images" / name)
        if isinstance(lf, str) and lf.strip():             # puceron -> label
            lsrc = labels_dir / lf
            if lsrc.exists():
                _link(lsrc, root / f"fold_{fold}" / "labels" / (Path(name).stem + ".txt"))
        ok += 1
    if verbose:
        print(f"CSV principal : {ok} tuiles placees, {miss} images introuvables")

    # --- Renfort "cell" -> _extra_train (ajoute au TRAIN uniquement) ---
    ex = variant.get("extra_train")
    if ex:
        (root / "_extra_train" / "images").mkdir(parents=True, exist_ok=True)
        (root / "_extra_train" / "labels").mkdir(parents=True, exist_ok=True)
        ex_df = pd.read_csv(ex["csv"])
        ex_imgs = build_image_index([ex["images_dir"]])
        ex_lbls = Path(ex["labels_dir"])
        n_ex = 0
        for name in ex_df["image"]:
            src_img = ex_imgs.get(name)
            if src_img is None:
                continue
            _link(src_img, root / "_extra_train" / "images" / name)
            lsrc = ex_lbls / (Path(name).stem + ".txt")
            if lsrc.exists():
                _link(lsrc, root / "_extra_train" / "labels" / (Path(name).stem + ".txt"))
            n_ex += 1
        if verbose:
            print(f"_extra_train (cell) : {n_ex} images ajoutees (train uniquement)")

    if verbose:
        check_folds()
    return root


def check_folds():
    """Compte les tuiles pucerons / fonds reellement placees par fold."""
    root = Path(cfg.FOLDS_ROOT)
    print("\nVerification (doit correspondre au CSV) :")
    for f in range(cfg.N_CV_FOLDS + 1):
        imgs = list((root / f"fold_{f}" / "images").glob("*"))
        npuc = sum(1 for im in imgs
                   if has_annotations(root / f"fold_{f}" / "labels" / f"{im.stem}.txt"))
        tag = " (test, non utilise en CV)" if f == cfg.TEST_FOLD else ""
        print(f"  fold {f} : {len(imgs)} images ({npuc} pucerons, {len(imgs)-npuc} fonds){tag}")


# ------------------------------------------------------- listes train figees
def rebase(line, root=None):
    """Replace une entree de liste figee sous la racine de folds courante.

    Les listes figees contiennent des chemins ABSOLUS, ecrits par la session qui
    les a creees (les notebooks d'origine utilisaient `/content/kfold_yolo/...`).
    Seule compte la queue du chemin -- `fold_2/images/x.jpg` ou
    `_extra_train/images/y.jpg` -- qui identifie la tuile : on la regreffe sur
    `cfg.FOLDS_ROOT`. La SELECTION de tuiles reste donc exactement celle qui a
    ete figee, quel que soit l'endroit ou les folds ont ete reconstruits.
    """
    root = Path(cfg.FOLDS_ROOT if root is None else root)
    parts = [p for p in re.split(r"[\\/]+", str(line).strip()) if p]
    if len(parts) >= 3:
        return str(root.joinpath(*parts[-3:]))
    return str(line).strip()


def read_frozen(txt_frozen, verbose=True):
    """Relit une liste figee et la regreffe sur la racine de folds courante."""
    lignes = [l for l in Path(txt_frozen).read_text().splitlines() if l.strip()]
    chemins = [rebase(l) for l in lignes]
    absentes = [p for p in chemins if not Path(p).exists()]
    if absentes:
        # Une tuile manquante change le jeu d'entrainement sans le dire : les
        # modeles ne verraient plus les memes donnees, ce qui vide de son sens
        # la liste figee. On refuse de continuer plutot que d'avertir.
        exemples = "\n  ".join(str(p) for p in absentes[:3])
        suite = f"\n  ... et {len(absentes) - 3} autres" if len(absentes) > 3 else ""
        raise RuntimeError(
            f"{len(absentes)} tuile(s) sur {len(chemins)} de la liste figee "
            f"{Path(txt_frozen).name} sont introuvables sous {cfg.FOLDS_ROOT}.\n"
            f"  {exemples}{suite}\n"
            "Les folds ont-ils ete construits dans cette session ? Lancer "
            "folds.build_folds(), puis folds.diagnose() pour le detail.")
    return chemins


def fold_train_paths(fold, neg_ratio=None, verbose=True):
    """Liste des tuiles de train du fold (4 autres folds + _extra_train).

    Les fonds sont sous-echantillonnes a `neg_ratio` x positifs. La liste est
    figee sur le Drive et relue aux appels suivants : elle est PARTAGEE par tous
    les modeles du benchmark, et par les notebooks d'origine.
    """
    neg_ratio = cfg.NEG_RATIO if neg_ratio is None else neg_ratio
    root = Path(cfg.FOLDS_ROOT)
    txt_frozen = Path(cfg.SPLITS_FROZEN) / f"train_fold{fold}_neg{neg_ratio}.txt"
    if txt_frozen.exists():
        return read_frozen(txt_frozen, verbose=verbose)

    pos, neg = [], []
    for i in range(cfg.N_CV_FOLDS):
        if i == fold:
            continue
        for img in sorted((root / f"fold_{i}" / "images").glob("*")):
            lbl = root / f"fold_{i}" / "labels" / f"{img.stem}.txt"
            (pos if has_annotations(lbl) else neg).append(str(img))
    ex = root / "_extra_train"
    if (ex / "images").exists():
        for img in sorted((ex / "images").glob("*")):
            lbl = ex / "labels" / f"{img.stem}.txt"
            (pos if has_annotations(lbl) else neg).append(str(img))

    random.seed(cfg.SEED + fold)
    negR = neg if len(neg) <= neg_ratio * len(pos) else random.sample(neg, neg_ratio * len(pos))
    lines = pos + negR
    txt_frozen.parent.mkdir(parents=True, exist_ok=True)
    txt_frozen.write_text("\n".join(lines))
    return lines


def fold_val_paths(fold):
    """Liste des tuiles de validation : le fold complet (positifs + fonds)."""
    root = Path(cfg.FOLDS_ROOT)
    return [str(p) for p in sorted((root / f"fold_{fold}" / "images").glob("*"))]


def fold_counts(fold, neg_ratio=None):
    """(n_pucerons_train, n_fonds_train) pour le fold."""
    paths = fold_train_paths(fold, neg_ratio)
    npos = sum(1 for p in paths if has_annotations(label_of(p)))
    return npos, len(paths) - npos


def diagnose(fold=0, neg_ratio=None):
    """Etat des folds et des listes figees : a lancer quand un fold semble vide.

    Repond aux trois questions qui bloquent en pratique : les folds sont-ils
    construits, les labels sont-ils bien lies, et les listes figees du Drive
    retombent-elles sur les tuiles de cette session ?
    """
    neg_ratio = cfg.NEG_RATIO if neg_ratio is None else neg_ratio
    root = Path(cfg.FOLDS_ROOT)
    print(f"Racine des folds : {root}")
    print(f"  existe : {root.exists()}\n")

    print("Contenu par fold (images / labels non vides) :")
    total_lbl = 0
    for f in range(cfg.N_CV_FOLDS + 1):
        imgs = list((root / f"fold_{f}" / "images").glob("*")) \
            if (root / f"fold_{f}" / "images").exists() else []
        npuc = sum(1 for im in imgs
                   if has_annotations(root / f"fold_{f}" / "labels" / f"{im.stem}.txt"))
        total_lbl += npuc
        print(f"  fold_{f:<2} {len(imgs):>6} images   {npuc:>6} avec label")
    ex = root / "_extra_train"
    if (ex / "images").exists():
        imgs = list((ex / "images").glob("*"))
        npuc = sum(1 for im in imgs if has_annotations(ex / "labels" / f"{im.stem}.txt"))
        total_lbl += npuc
        print(f"  _extra_train {len(imgs):>3} images   {npuc:>6} avec label")
    if total_lbl == 0:
        print("\n  ATTENTION : aucun label lie. Verifier labels_dir et la colonne "
              "label_file du CSV de split, puis relancer folds.build_folds().")

    txt = Path(cfg.SPLITS_FROZEN) / f"train_fold{fold}_neg{neg_ratio}.txt"
    print(f"\nListe figee du fold {fold} : {txt}")
    if not txt.exists():
        print("  absente : elle sera calculee et ecrite au premier appel.")
        return
    lignes = [l for l in txt.read_text().splitlines() if l.strip()]
    bruts = sum(1 for l in lignes if Path(l.strip()).exists())
    rebases = [rebase(l) for l in lignes]
    ok = sum(1 for p in rebases if Path(p).exists())
    print(f"  {len(lignes)} entrees | {bruts} valides telles quelles | "
          f"{ok} apres regreffe sur la racine courante")
    if lignes:
        print(f"  exemple d'entree figee : {lignes[0]}")
        print(f"  regreffee              : {rebases[0]}")
    if ok:
        npos = sum(1 for p in rebases if Path(p).exists() and has_annotations(label_of(p)))
        print(f"  dont {npos} tuiles annotees et {ok - npos} fonds")
