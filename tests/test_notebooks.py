# -*- coding: utf-8 -*-
"""Valide les notebooks du depot.

    python tests/test_notebooks.py

Editer un `.ipynb` a la main dans un editeur de texte casse facilement le JSON :
un guillemet non echappe dans une ligne de code (`"REPO_DIR = "/content/x""`)
suffit a rendre le fichier illisible par Colab, qui annonce alors un "notebook
invalide". Ce test attrape ce cas avant qu'il ne parte dans le depot :

  - le fichier est du JSON valide et respecte le schema nbformat ;
  - chaque cellule de code sans magie IPython (`!`, `%`) est du Python compilable ;
  - les notebooks sont conformes a ce que produit tools/make_notebooks.py,
    qui reste la source de verite.
"""

import json
import subprocess
import sys
from pathlib import Path

RACINE = Path(__file__).resolve().parent.parent
NB_DIR = RACINE / "notebooks"


def cellules_code(doc):
    for c in doc["cells"]:
        if c["cell_type"] == "code":
            yield "".join(c["source"])


def main():
    fichiers = sorted(NB_DIR.glob("*.ipynb"))
    if not fichiers:
        raise SystemExit(f"Aucun notebook dans {NB_DIR}")

    try:
        import nbformat
    except ImportError:
        nbformat = None
        print("nbformat absent : validation JSON et Python seulement\n")

    for f in fichiers:
        # 1. JSON valide (le cas qui casse Colab)
        try:
            doc = json.loads(f.read_text(encoding="utf-8"))
        except json.JSONDecodeError as e:
            ligne = f.read_text(encoding="utf-8").splitlines()[max(0, e.lineno - 1)]
            raise SystemExit(f"{f.name} : JSON invalide ligne {e.lineno} ({e.msg})\n"
                             f"  -> {ligne.strip()[:120]}\n"
                             f"  Regenerer avec : python tools/make_notebooks.py")

        # 2. Schema nbformat
        if nbformat is not None:
            nbformat.validate(nbformat.read(str(f), as_version=4))

        # 3. Cellules de code compilables (hors magie IPython)
        n_py = 0
        for src in cellules_code(doc):
            if any(l.lstrip().startswith(("!", "%")) for l in src.splitlines()):
                continue
            try:
                compile(src, f"{f.name}:cellule", "exec")
            except SyntaxError as e:
                raise SystemExit(f"{f.name} : cellule non compilable ligne {e.lineno} "
                                 f": {e.msg}\n  -> {(e.text or '').strip()[:120]}")
            n_py += 1
        print(f"{f.name:48s} OK  {len(doc['cells']):2d} cellules "
              f"({n_py} cellules Python verifiees)")

    # 4. Conformite au generateur : les notebooks sont-ils a jour ?
    avant = {f: f.read_bytes() for f in fichiers}
    subprocess.run([sys.executable, str(RACINE / "tools" / "make_notebooks.py")],
                   capture_output=True, text=True, check=True)
    modifies = [f.name for f, contenu in avant.items() if f.read_bytes() != contenu]
    if modifies:
        raise SystemExit(
            "\nNotebooks desynchronises de tools/make_notebooks.py : "
            + ", ".join(modifies)
            + "\nIls viennent d'etre regeneres : verifier le diff avant de commiter.")
    print("\nTOUS LES NOTEBOOKS SONT VALIDES ET A JOUR")


if __name__ == "__main__":
    main()
