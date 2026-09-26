# -*- coding: utf-8 -*-
"""Lecture des journaux d'entrainement, et early stopping (hors service).

LES LECTEURS SONT EN SERVICE. `lire_log_json` (RT-DETR, D-FINE),
`lire_metrics_csv` (RF-DETR) et `lire_log_yolox` (YOLOX) extraient la metrique
de validation native de chaque framework, epoque par epoque. Les runners s'en
servent pour renseigner `best_epoch` et `epochs_run`, et `detr_repo` pour
departager `best_stg1.pth` et `best_stg2.pth`.

L'EARLY STOPPING N'EST PLUS UTILISE. Le benchmark donne desormais le meme
budget fixe (`cfg.MAX_EPOCHS`) aux sept modeles, chacun designant ensuite son
meilleur checkpoint sur sa metrique native. `Surveillant` et
`SurveillanceJournal` restent ici, fonctionnels et testes, pour un usage futur :
aucun runner ne les instancie.

La metrique lue est toujours la metrique NATIVE du framework, jamais
l'evaluateur unifie d'aphids_det.evaluate, qui coute bien trop cher pour etre
relance a chaque epoque. La comparaison finale entre modeles, elle, reste faite
par cet evaluateur unifie.
"""

import json
import re
import threading
import time
from pathlib import Path


class Surveillant:
    """Suit une metrique a maximiser et dit quand arreter.

    `observer(valeurs)` recoit la liste complete des valeurs lues depuis le
    debut ; seules les nouvelles sont prises en compte. Renvoie True des que
    `patience` epoques se sont ecoulees sans amelioration superieure a
    `min_delta`.
    """

    def __init__(self, patience, min_delta=0.0, nom="modele"):
        self.patience = patience
        self.min_delta = min_delta
        self.nom = nom
        self.meilleure = None
        self.meilleure_epoque = -1
        self.vues = 0
        self.declenche = False

    def observer(self, valeurs):
        for valeur in valeurs[self.vues:]:
            self.vues += 1
            if self.meilleure is None or valeur > self.meilleure + self.min_delta:
                self.meilleure = valeur if self.meilleure is None else max(
                    valeur, self.meilleure)
                self.meilleure_epoque = self.vues
                print(f"    [{self.nom}] epoque {self.vues} : {valeur:.4f} "
                      f"(nouveau maximum)", flush=True)
            elif self.vues - self.meilleure_epoque >= self.patience:
                self.declenche = True
                print(f"    [{self.nom}] arret anticipe a l'epoque {self.vues} : "
                      f"{self.patience} epoques sans gain > {self.min_delta} "
                      f"(meilleure : {self.meilleure:.4f} a l'epoque "
                      f"{self.meilleure_epoque})", flush=True)
        return self.declenche

    def resume(self):
        return {"best_epoch": self.meilleure_epoque, "epochs_run": self.vues,
                "stopped_early": self.declenche, "meilleure": self.meilleure}


# ------------------------------------------------------------------ lecteurs
def lire_log_json(journal, cle="test_coco_eval_bbox", indice=0):
    """Valeurs par epoque d'un `log.txt` JSON par ligne (RT-DETR, D-FINE, RF-DETR)."""
    journal = Path(journal)
    if not journal.exists():
        return []
    valeurs = []
    for ligne in journal.read_text(errors="replace").splitlines():
        try:
            d = json.loads(ligne)
        except Exception:
            continue
        stats = d.get(cle)
        if isinstance(stats, (list, tuple)) and len(stats) > indice:
            valeurs.append(float(stats[indice]))
        elif isinstance(stats, (int, float)):
            valeurs.append(float(stats))
    return valeurs


# `summarize()` de pycocotools ecrit 12 lignes par evaluation, dont 6 "Average
# Recall" qui partagent la plage IoU=0.50:0.95 et l'aire "all". Sans le prefixe
# "Average Precision" et sans maxDets=100, la regex en attrapait 4 par epoque :
# la patience etait divisee par 4.
_AP_YOLOX = re.compile(
    r"Average Precision\s+\(AP\)\s*@\[\s*IoU=0\.50:0\.95\s*\|\s*area=\s*all\s*"
    r"\|\s*maxDets=\s*100\s*\]\s*=\s*([-\d.]+)")


def lire_log_yolox(journal):
    """mAP50-95 par evaluation, lues dans le `train_log.txt` de YOLOX."""
    journal = Path(journal)
    if not journal.exists():
        return []
    return [float(v) for v in _AP_YOLOX.findall(journal.read_text(errors="replace"))
            if float(v) >= 0]


def lire_metrics_csv(journal, colonnes=("val/ema_mAP_50_95", "val/mAP_50_95")):
    """mAP50-95 par epoque, lues dans le `metrics.csv` de RF-DETR.

    La branche PyTorch Lightning de rfdetr journalise par `CSVLogger(save_dir=
    output_dir, name="", version="")`, qui ecrit donc `<output_dir>/metrics.csv`.
    Les colonnes utiles sont `val/ema_mAP_50_95` et `val/mAP_50_95` (cf.
    rfdetr/training/callbacks/coco_eval.py) ; la premiere disponible est prise,
    l'EMA d'abord puisque c'est `checkpoint_best_ema.pth` qui est evalue.
    """
    import csv

    journal = Path(journal)
    if not journal.exists():
        return []
    with journal.open(newline="") as f:
        lignes = list(csv.DictReader(f))
    for colonne in colonnes:
        valeurs = [float(l[colonne]) for l in lignes
                   if l.get(colonne) not in (None, "")]
        if valeurs:
            return valeurs
    return []


# ------------------------------------------------------------- surveillance
class SurveillanceJournal:
    """Scrute un journal en tache de fond et arrete le processus le moment venu.

    Passee a `external.run`, qui appelle `demarrer(proc)` puis `arreter()`.
    """

    def __init__(self, journal, lecteur, surveillant, intervalle=20.0):
        self.journal = Path(journal)
        self.lecteur = lecteur
        self.surveillant = surveillant
        self.intervalle = intervalle
        self._fil = None
        self._fini = threading.Event()
        self.interrompu = False

    def demarrer(self, proc):
        def boucle():
            while not self._fini.wait(self.intervalle):
                try:
                    if self.surveillant.observer(self.lecteur(self.journal)):
                        self.interrompu = True
                        print("    arret du processus d'entrainement...", flush=True)
                        proc.terminate()
                        try:
                            proc.wait(timeout=60)
                        except Exception:
                            proc.kill()
                        return
                except Exception as e:                  # jamais fatal
                    print(f"    surveillance : {type(e).__name__}: {e}", flush=True)

        self._fil = threading.Thread(target=boucle, daemon=True)
        self._fil.start()

    def arreter(self):
        self._fini.set()
        if self._fil is not None:
            self._fil.join(timeout=5)
        # derniere lecture : le journal a pu s'enrichir apres le dernier tour
        try:
            self.surveillant.observer(self.lecteur(self.journal))
        except Exception:
            pass
