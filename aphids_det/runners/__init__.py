# -*- coding: utf-8 -*-
"""Runners par framework.

Un runner = installer le framework, entrainer un fold, exporter les predictions
au format COCO, puis deleguer les metriques a aphids_det.evaluate.

Les imports sont volontairement paresseux : chaque notebook n'installe QUE son
framework (leurs dependances sont incompatibles entre elles), donc importer
tous les runners d'un coup echouerait.
"""

__all__ = ["ultralytics_runner", "rfdetr_runner", "rtdetr_runner",
           "dfine_runner", "yolox_runner", "detr_repo"]
