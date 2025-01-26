import os
import typing

import numpy as np
import torch
from torch.hub import load_state_dict_from_url

from models.dsfd.base import Detector
from models.dsfd.config import resnet152_model_config
from models.dsfd.face_ssd import SSD
from models.dsfd.torch_utils import *

model_url = "https://api.loke.aws.unit.no/dlr-gui-backend-resources-content/v2/contents/links/61be4ec7-8c11-4a4a-a9f4-827144e4ab4f0c2764c1-80a0-4083-bbfa-68419f889b80e4692358-979b-458e-97da-c1a1660b3314"


class DSFDDetector(Detector):

    def __init__(
            self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        model_path = kwargs.get('model_path', None)
        if model_path and os.path.exists(model_path):
            state_dict = torch.load(model_path, map_location=self.device, weights_only=True)
        else:
            state_dict = load_state_dict_from_url(
            model_url,
            map_location=self.device,
            progress=True)
        self.net = SSD(resnet152_model_config)
        self.net.load_state_dict(state_dict)
        self.net.eval()
        self.net = self.net.to(self.device)

    @torch.no_grad()
    def _detect(self, x: torch.Tensor,) -> typing.List[np.ndarray]:
        """Batched detect
        Args:
            image (np.ndarray): shape [N, H, W, 3]
        Returns:
            boxes: list of length N with shape [num_boxes, 5] per element
        """
        # Expects BGR
        x = x[:, [2, 1, 0], :, :]
        with torch.cuda.amp.autocast(enabled=self.fp16_inference):
            boxes = self.net(
                x, self.confidence_threshold, self.nms_iou_threshold
            )
        return boxes
