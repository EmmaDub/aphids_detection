"""Verifie qu'une liste figee ecrite par les notebooks d'origine
(/content/kfold_yolo/...) est relue correctement quand les folds vivent
ailleurs (/content/aphids_work/folds/...).

    python tests/test_rebase.py
"""
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import aphids_det.config as cfg

tmp = Path(tempfile.mkdtemp())
cfg.WORK_ROOT = tmp / "work"
cfg.OUT_DIR = tmp / "out"
cfg.refresh()

from aphids_det import folds  # noqa: E402

root = Path(cfg.FOLDS_ROOT)
# folds "reconstruits" : 2 tuiles annotees + 1 fond dans fold_1, 1 renfort cell
for sous, nom, avec_label in [("fold_1", "p0.jpg", True), ("fold_1", "p1.jpg", True),
                              ("fold_1", "bg0.jpg", False),
                              ("_extra_train", "cell0.jpg", True)]:
    (root / sous / "images").mkdir(parents=True, exist_ok=True)
    (root / sous / "labels").mkdir(parents=True, exist_ok=True)
    (root / sous / "images" / nom).write_bytes(b"x")
    if avec_label:
        (root / sous / "labels" / (Path(nom).stem + ".txt")).write_text("0 0.5 0.5 0.1 0.1\n")

# liste figee telle que l'ecrivaient les notebooks d'origine (chemins absolus
# d'une AUTRE session, avec un separateur POSIX)
ANCIENNE_RACINE = "/content/kfold_yolo/yolo_neg1/labels_cell_20"
frozen = Path(cfg.SPLITS_FROZEN) / f"train_fold0_neg{cfg.NEG_RATIO}.txt"
frozen.parent.mkdir(parents=True, exist_ok=True)
COMPLETE = [
    f"{ANCIENNE_RACINE}/fold_1/images/p0.jpg",
    f"{ANCIENNE_RACINE}/fold_1/images/p1.jpg",
    f"{ANCIENNE_RACINE}/_extra_train/images/cell0.jpg",
    f"{ANCIENNE_RACINE}/fold_1/images/bg0.jpg",
]
frozen.write_text("\n".join(COMPLETE))

chemins = folds.fold_train_paths(0)
print(f"\n{len(chemins)} tuiles retrouvees :")
for p in chemins:
    print("   ", p)
assert len(chemins) == 4, chemins
assert all(Path(p).exists() for p in chemins), "chemins non resolus"
assert all(str(root) in p for p in chemins), "regreffe non appliquee"

npos, nneg = folds.fold_counts(0)
print(f"\nfold_counts -> {npos} pucerons / {nneg} fonds (attendu 3 / 1)")
assert (npos, nneg) == (3, 1), (npos, nneg)

# la selection figee est respectee : l'ordre et le contenu suivent le fichier
assert Path(chemins[0]).name == "p0.jpg" and Path(chemins[-1]).name == "bg0.jpg"

# entree Windows (antislash) : meme resultat
assert folds.rebase(r"C:\autre\racine\fold_1\images\p0.jpg") == str(
    root / "fold_1" / "images" / "p0.jpg")

# --- une tuile manquante doit ARRETER le benchmark, pas passer inapercue ---
# Sans cela, les modeles ne s'entrainent plus sur les memes donnees et la liste
# figee ne garantit plus rien.
frozen.write_text("\n".join(COMPLETE + [f"{ANCIENNE_RACINE}/fold_1/images/disparue.jpg"]))
try:
    folds.fold_train_paths(0)
except RuntimeError as e:
    message = str(e)
    print(f"\ntuile manquante -> RuntimeError : {message.splitlines()[0]}")
    assert "1 tuile(s) sur 5" in message, message
    assert "disparue.jpg" in message, message
    assert "folds.diagnose()" in message, message
else:
    raise AssertionError("une tuile manquante doit lever une RuntimeError")
frozen.write_text("\n".join(COMPLETE))     # on remet la liste complete

print("\n--- diagnose() ---")
folds.diagnose(0)
print("\nTEST REBASE OK")
