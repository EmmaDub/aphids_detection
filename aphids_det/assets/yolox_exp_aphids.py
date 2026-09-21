# -*- coding: utf-8 -*-
"""Fichier d'experience YOLOX-Nano pour le benchmark pucerons.

Copie dans `exps/aphids/` du depot YOLOX clone. Les parametres du fold sont lus
dans le JSON de meme nom, ecrit par aphids_det/runners/yolox_runner.py : rien
n'est code en dur ici, le meme fichier sert aux 5 folds.

La classe `Exp` reprend exactement celle de `exps/default/yolox_nano.py`
(profondeur 0.33, largeur 0.25, tete depthwise) ; seuls changent le dataset, la
resolution (640 pour rester comparable aux autres modeles), le budget d'epoques
et les augmentations.

Deux ecarts a YOLOX par defaut, imposes par la reference d'augmentation :
  - miroir vertical (flipud = 0.5), absent de YOLOX ;
  - gains HSV regles sur hsv_s / hsv_v = 0.2 au lieu des valeurs internes.
Les deux sont apportes par `AphidsTrainTransform`, substitue a
`yolox.data.TrainTransform` (que `Exp.get_data_loader` resout a l'execution).
"""

import json
import random
from pathlib import Path

import torch.nn as nn

from yolox.exp import Exp as MyExp

P = json.loads(Path(__file__).with_suffix(".json").read_text())


# --------------------------------------------------------------------------
# TrainTransform enrichi : flip vertical + HSV aux gains du benchmark
# --------------------------------------------------------------------------
import yolox.data as _yolox_data                                    # noqa: E402
from yolox.data.data_augment import TrainTransform as _TrainTransform  # noqa: E402
from yolox.data.data_augment import augment_hsv as _augment_hsv     # noqa: E402


class AphidsTrainTransform(_TrainTransform):
    """TrainTransform de YOLOX + miroir vertical et gains HSV explicites."""

    def __init__(self, *args, **kwargs):
        self._hsv_prob = kwargs.get("hsv_prob", 1.0)
        kwargs["hsv_prob"] = 0.0          # l'HSV est applique ici, pas dans le parent
        super().__init__(*args, **kwargs)

    def __call__(self, image, targets, input_dim):
        if P["vflip_prob"] > 0 and random.random() < P["vflip_prob"]:
            image = image[::-1].copy()                     # image HWC
            h = image.shape[0]
            if len(targets):
                y1 = targets[:, 1].copy()
                y2 = targets[:, 3].copy()
                targets[:, 1] = h - y2                     # boites en xyxy
                targets[:, 3] = h - y1
        if self._hsv_prob > 0 and random.random() < self._hsv_prob:
            hg, sg, vg = P["hsv_gains"]
            _augment_hsv(image, hgain=hg, sgain=sg, vgain=vg)
        return super().__call__(image, targets, input_dim)


_yolox_data.TrainTransform = AphidsTrainTransform


# --------------------------------------------------------------------------
class Exp(MyExp):
    def __init__(self):
        super(Exp, self).__init__()
        # --- architecture : identique a exps/default/yolox_nano.py ---
        self.depth = 0.33
        self.width = 0.25

        # --- donnees ---
        self.data_dir = P["data_dir"]
        self.train_ann = P["train_ann"]
        self.val_ann = P["val_ann"]
        self.num_classes = P["num_classes"]
        self.data_num_workers = P["num_workers"]

        # --- resolution : 640 (et non 416) pour rester comparable ---
        self.input_size = (P["imgsz"], P["imgsz"])
        self.test_size = (P["imgsz"], P["imgsz"])
        self.random_size = None
        self.multiscale_range = 0          # pas de multi-echelle

        # --- budget ---
        self.seed = P["seed"]              # YOLOX lit la graine ici, pas en CLI
        self.max_epoch = P["epochs"]
        self.no_aug_epochs = P["no_aug_epochs"]
        self.eval_interval = 1
        self.print_interval = 50
        self.output_dir = P["output_dir"]

        # --- augmentation (reference du benchmark) ---
        self.mosaic_prob = P["mosaic_prob"]
        self.mixup_prob = P["mixup_prob"]
        self.enable_mixup = P["enable_mixup"]
        self.hsv_prob = P["hsv_prob"]
        self.flip_prob = P["flip_prob"]
        self.degrees = P["degrees"]
        self.translate = P["translate"]
        self.mosaic_scale = tuple(P["mosaic_scale"])
        self.shear = P["shear"]

        self.exp_name = P["exp_name"]

    def get_model(self, sublinear=False):
        def init_yolo(M):
            for m in M.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.eps = 1e-3
                    m.momentum = 0.03

        if "model" not in self.__dict__:
            from yolox.models import YOLOX, YOLOPAFPN, YOLOXHead
            in_channels = [256, 512, 1024]
            # NANO : depthwise = True, c'est la difference principale.
            backbone = YOLOPAFPN(self.depth, self.width, in_channels=in_channels,
                                 act=self.act, depthwise=True)
            head = YOLOXHead(self.num_classes, self.width, in_channels=in_channels,
                             act=self.act, depthwise=True)
            self.model = YOLOX(backbone, head)

        self.model.apply(init_yolo)
        self.model.head.initialize_biases(1e-2)
        return self.model
