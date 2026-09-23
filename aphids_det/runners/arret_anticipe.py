# -*- coding: utf-8 -*-
"""Early stopping pour les depots qui n'en ont pas : RT-DETR, D-FINE, YOLOX.

Ultralytics et RF-DETR arretent d'eux-memes. Les trois autres consomment tout
leur plafond d'epoques. Comme ils s'entrainent dans un sous-processus, on ne
peut pas y brancher un callback Python : on surveille donc le journal que le
framework ecrit deja, epoque par epoque, et on met fin au processus quand la
patience est epuisee.

La metrique lue est la metrique de validation NATIVE du framework (mAP50-95 de
sa propre evaluation COCO), jamais l'evaluateur unifie d'aphids_det.evaluate :
le relancer a chaque epoque couterait bien plus cher que l'entrainement. La
comparaison finale entre modeles, elle, reste faite par l'evaluateur unifie.

Les poids de la meilleure epoque ne sont pas geres ici : chaque depot sauvegarde
deja son meilleur checkpoint (`best.pth`, `best_stg2.pth`, `best_ckpt.pth`), et
`detr_repo.best_checkpoint` / `yolox_runner` les retrouvent. Arreter le
processus apres la meilleure epoque laisse donc ces fichiers en place.
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


_AP_YOLOX = re.compile(
    r"IoU=0\.50:0\.95\s*\|\s*area=\s*all\s*\|\s*maxDets=\s*\d+\s*\]\s*=\s*([-\d.]+)")


def lire_log_yolox(journal):
    """mAP50-95 par evaluation, lues dans le `train_log.txt` de YOLOX."""
    journal = Path(journal)
    if not journal.exists():
        return []
    return [float(v) for v in _AP_YOLOX.findall(journal.read_text(errors="replace"))
            if float(v) >= 0]


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
