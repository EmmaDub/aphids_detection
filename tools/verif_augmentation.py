# -*- coding: utf-8 -*-
"""Les trois verifications exigees apres chaque modification d'augmentation.

    python tools/verif_augmentation.py --modele ultralytics
    python tools/verif_augmentation.py --modele tous

Pour chaque modele demande :
  1. dump visuel de 18 tuiles augmentees, boites dessinees, sur des fichiers
     reels du jeu de donnees ;
  2. impression du pipeline d'augmentation RESOLU -- l'objet transform construit
     par le framework, pas le fichier de configuration ;
  3. controle de syntaxe de bout en bout (import du package, compilation de tous
     ses modules, validation des notebooks).

A lancer dans le runtime qui contient le framework concerne : chaque modele a
besoin du sien, et les quatre ne cohabitent pas. Le script n'echoue jamais en
silence : toute verification manquee sort en code 1.
"""

import argparse
import py_compile
import sys
import traceback
from pathlib import Path

RACINE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RACINE))

import aphids_det.config as cfg  # noqa: E402
from aphids_det import augment, visualize  # noqa: E402

MODELES = {
    "ultralytics": augment.MODELES["ultralytics"],
    "rfdetr": augment.MODELES["rfdetr"],
    "detr": augment.MODELES["detr"],
    "yolox": augment.MODELES["yolox"],
}


# ------------------------------------------------------- pipeline resolu
def pipeline_ultralytics():
    """Compose construit par Ultralytics, tel que le dataset l'utilise."""
    from ultralytics.data.augment import Albumentations

    tuiles, vivier = visualize.sample_tiles(fold=0, n=1, seed=0, min_boxes=1)
    ds, _idx = visualize._ultralytics_dataset(
        tuiles[0], vivier, {k: v for k, v in augment.AUG.items()})
    bloc = Albumentations(p=1.0)
    return {
        "transforms (objet construit)": ds.transforms,
        "bloc Albumentations cache": f"transform={bloc.transform!r} "
                                     f"(None = neutralise)",
        "hyp effectifs": {k: getattr(ds.hyp, k, "?") for k in augment.AUG},
    }


def pipeline_yolox():
    """MosaicDetection + TrainTransform construits par l'exp YOLOX."""
    from aphids_det.runners import yolox_runner
    from yolox.exp import get_exp

    repo, _commit = yolox_runner.setup(install=False)
    from aphids_det import cocoify
    ds, _p, _n = cocoify.build_coco_fold(0, layout="coco", verbose=False)
    sortie = Path(cfg.WORK_ROOT) / "runs" / "yolox"
    exp_py = yolox_runner.write_exp(repo, 0, ds, sortie)
    exp = get_exp(str(exp_py), None)

    import yolox.data as yolox_data
    from yolox.data import data_augment as da
    from yolox.data.datasets import mosaicdetection as md
    return {
        "attributs d'augmentation de l'Exp": {
            k: getattr(exp, k, "absent") for k in
            ("mosaic_prob", "mosaic_scale", "scale", "degrees", "translate",
             "shear", "perspective", "flip_prob", "hsv_prob", "enable_mixup",
             "mixup_prob", "multiscale_range", "no_aug_epochs", "input_size")},
        "TrainTransform effectif": yolox_data.TrainTransform,
        "random_affine effectif": (da.random_affine.__name__,
                                   md.random_affine.__name__),
        "HSV": "augment_hsv_multiplicatif (gain x[0.8, 1.2], p=1)",
    }


def pipeline_rfdetr():
    """Compose construit par RF-DETR pour le split train."""
    from rfdetr.datasets.coco import make_coco_transforms

    tf = make_coco_transforms(
        image_set="train", resolution=cfg.IMGSZ, aug_config=augment.AUG_RFDETR,
        **{k: v for k, v in augment.RFDETR_COUPURES.items()
           if k in make_coco_transforms.__code__.co_varnames})
    return {"transforms (objet construit)": tf,
            "aug_config passe": augment.AUG_RFDETR,
            "coupures demandees": augment.RFDETR_COUPURES}


def pipeline_detr(variante="rtdetr_v2"):
    """Compose construit par le depot RT-DETR / D-FINE depuis notre yml."""
    import os

    from aphids_det import cocoify
    from aphids_det.runners import detr_repo

    _racine, work, _commit = detr_repo.setup(variante, install=False)
    ds, _p, _n = cocoify.build_coco_fold(0, layout="coco", verbose=False)
    sortie = Path(cfg.WORK_ROOT) / "runs" / f"{variante}_verif"
    batch, _ = cfg.batch_for(detr_repo.SPECS[variante]["framework"])
    chemin = detr_repo.write_config(variante, 0, ds, sortie, batch, work)

    detr_repo.external.add_to_path(work)
    cwd = os.getcwd()
    os.chdir(work)
    try:
        from src.core import YAMLConfig
        conf = YAMLConfig(str(chemin))
        transforms = conf.train_dataloader.dataset.transforms
    finally:
        os.chdir(cwd)
    return {"transforms (objet construit)": transforms,
            "config generee": chemin}


PIPELINES = {"ultralytics": pipeline_ultralytics, "yolox": pipeline_yolox,
             "rfdetr": pipeline_rfdetr, "detr": pipeline_detr}


# --------------------------------------------------------- verifications
def verif_syntaxe():
    """Controle de syntaxe de bout en bout : modules, puis notebooks."""
    fichiers = sorted(RACINE.glob("aphids_det/**/*.py")) + \
        sorted(RACINE.glob("tools/*.py")) + sorted(RACINE.glob("tests/*.py"))
    for f in fichiers:
        py_compile.compile(str(f), doraise=True)
    print(f"  syntaxe : {len(fichiers)} modules compiles")

    import json
    nbs = sorted(RACINE.glob("notebooks/*.ipynb"))
    for nb in nbs:
        json.loads(nb.read_text(encoding="utf-8"))
    print(f"  syntaxe : {len(nbs)} notebooks valides")
    return True


def verif_dimensions(fold=0, n=40):
    """Dimensions reelles des tuiles : un resize carre etire-t-il le ratio ?

    RT-DETR, D-FINE et RF-DETR redimensionnent en carre ; si les tuiles ne sont
    pas carrees, ils deforment la morphologie des pucerons la ou Ultralytics et
    YOLOX preservent le ratio par letterbox. Le correctif (pre-pad a 114 en
    amont) n'est PAS applique ici : il demande une validation separee.
    """
    from PIL import Image

    from aphids_det.folds import fold_train_paths

    tailles = {}
    for p in fold_train_paths(fold)[:n]:
        with Image.open(p) as im:
            tailles[im.size] = tailles.get(im.size, 0) + 1
    print(f"  dimensions relevees sur {sum(tailles.values())} tuiles : "
          + ", ".join(f"{w}x{h} ({c})" for (w, h), c in sorted(tailles.items())))
    carrees = all(w == h for w, h in tailles)
    print(f"  tuiles carrees : {'oui -- aucun etirement de ratio' if carrees else 'NON'}")
    if not carrees:
        print("  ATTENTION : le resize carre de RT-DETR, D-FINE et RF-DETR etire "
              "le ratio, contrairement au letterbox d'Ultralytics et de YOLOX. "
              "Correctif de pre-pad a decider separement.")
    if tailles and cfg.TILE_SIZE not in {w for w, _ in tailles}:
        print(f"  ATTENTION : cfg.TILE_SIZE = {cfg.TILE_SIZE} ne correspond pas "
              f"aux dimensions relevees -- la conversion COCO suppose cette taille.")
    return carrees


def verifier(cle, n=18, fold=0):
    """Les trois verifications pour un modele. Renvoie True si toutes passent."""
    libelle = MODELES[cle]
    print(f"\n{'=' * 70}\n{libelle}\n{'=' * 70}")
    ok = True

    print("\n[2] pipeline d'augmentation resolu")
    try:
        for titre, valeur in PIPELINES[cle]().items():
            print(f"  --- {titre} ---")
            print("  " + str(valeur).replace("\n", "\n  "))
    except Exception:
        ok = False
        print("  ECHEC :")
        traceback.print_exc()

    print(f"\n[1] dump visuel de {n} tuiles augmentees")
    try:
        visualize.dump_augmentations(libelle, n=n, fold=fold)
    except Exception:
        ok = False
        print("  ECHEC :")
        traceback.print_exc()

    print("\n[3] controle de syntaxe de bout en bout")
    try:
        verif_syntaxe()
    except Exception:
        ok = False
        print("  ECHEC :")
        traceback.print_exc()

    print(f"\n-> {libelle} : {'TOUTES LES VERIFS PASSENT' if ok else 'VERIFS EN ECHEC'}")
    return ok


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--modele", default="tous",
                    choices=[*MODELES, "tous"],
                    help="modele a verifier (defaut : tous ceux du runtime)")
    ap.add_argument("--n", type=int, default=18, help="tuiles du dump visuel")
    ap.add_argument("--fold", type=int, default=0)
    args = ap.parse_args()

    cfg.refresh()
    print("=" * 70)
    print("Dimensions des tuiles d'entree")
    print("=" * 70)
    try:
        verif_dimensions(fold=args.fold)
    except Exception:
        print("  releve impossible :")
        traceback.print_exc()

    cibles = list(MODELES) if args.modele == "tous" else [args.modele]
    resultats = {c: verifier(c, n=args.n, fold=args.fold) for c in cibles}

    print(f"\n{'=' * 70}")
    for c, ok in resultats.items():
        print(f"  {MODELES[c]:32s} {'OK' if ok else 'ECHEC'}")
    sys.exit(0 if all(resultats.values()) else 1)


if __name__ == "__main__":
    main()
