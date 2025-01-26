import numpy as np
import torch
import torch.nn.functional as F
import torchvision
from torchvision import transforms
from einops import rearrange
import typing
from abc import ABC, abstractmethod
from torchvision.ops import nms

from models.dsfd.box_utils import scale_boxes


def check_image(im: np.ndarray):
    assert im.dtype == torch.Tensor,\
        f"Expect image to have dtype torch.Tensor. Was: {im.dtype}"
    assert len(im.shape) == 4,\
        f"Expected image to have 4 dimensions. got: {im.shape}"
    assert im.shape[1] == 3,\
        f"Expected image to be RGB, got: {im.shape[1]} color channels"


class Detector(ABC):
    def __init__(
            self,
            confidence_threshold: float,
            nms_iou_threshold: float,
            device: torch.device,
            max_resolution: int,
            fp16_inference: bool,
            clip_boxes: bool,
            model_path: str):
        """
        Args:
            confidence_threshold (float): Threshold to filter out bounding boxes
            nms_iou_threshold (float): Intersection over union threshold for non-maxima threshold
            device ([type], optional): Defaults to cuda if cuda capable device is available.
            max_resolution (int, optional): Max image resolution to do inference to.
            fp16_inference: To use torch autocast for fp16 inference or not
            clip_boxes: To clip boxes within [0,1] to ensure no boxes are outside the image
        """
        self.confidence_threshold = confidence_threshold
        self.nms_iou_threshold = nms_iou_threshold
        self.device = device
        self.max_resolution = max_resolution
        self.fp16_inference = fp16_inference
        self.clip_boxes = clip_boxes
        self.model_path = model_path

    def process_video(self, video_tensor: torch.Tensor, shrink) -> torch.Tensor:
        video_tensor = video_tensor.float() / 255.0
        video_tensor = rearrange(video_tensor, 't h w c -> t c h w')
        if shrink != 1.0:
            video_tensor = F.interpolate(video_tensor, scale_factor=shrink, mode='bilinear', align_corners=False)
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        return (video_tensor - mean) * 255.0

    @torch.no_grad()
    def detect(
        self, video_tensor: torch.Tensor, shrink=1.0) -> typing.List[np.ndarray]:
        """Takes N RGB image and performs and returns a set of bounding boxes as
            detections
        Args:
            image (torch.Tensor): shape [N, height, width, 3]
        Returns:
            np.ndarray: a list with N set of bounding boxes of
                shape [B, 5] with (xmin, ymin, xmax, ymax, score)
        """
        if video_tensor.dim() == 3:
            video_tensor = video_tensor.unsqueeze(0)
        video_tensor = self.process_video(video_tensor, shrink)
        video_tensor = video_tensor.to(self.device)
        boxes = self._batched_detect(video_tensor)
        height, width = video_tensor.shape[2], video_tensor.shape[3]
        boxes = [scale_boxes((height, width), box).cpu().numpy() for box in boxes]
        self.validate_detections(boxes)
        return boxes

    @abstractmethod
    def _detect(self, image: torch.Tensor) -> torch.Tensor:
        """Takes N RGB image and performs and returns a set of bounding boxes as
            detections
        Args:
            image (torch.Tensor): shape [N, 3, height, width]
        Returns:
            torch.Tensor: of shape [N, B, 5] with (xmin, ymin, xmax, ymax, score)
        """
        raise NotImplementedError

    def _batched_detect(self, image: torch.Tensor) -> typing.List[np.ndarray]:
        torch.save(image.cpu(), "test.pt")
        boxes = self._detect(image)
        boxes = self.filter_boxes(boxes)
        if self.clip_boxes:
            boxes = [box.clamp(0, 1) for box in boxes]
        return boxes

    def filter_boxes(self, boxes: torch.Tensor) -> typing.List[np.ndarray]:
        """Performs NMS and score thresholding

        Args:
            boxes (torch.Tensor): shape [N, B, 5] with (xmin, ymin, xmax, ymax, score)
        Returns:
            list: N np.ndarray of shape [B, 5]
        """
        final_output = []
        for i in range(len(boxes)):
            scores = boxes[i, :,  4]
            keep_idx = scores >= self.confidence_threshold
            boxes_ = boxes[i, keep_idx, :-1]
            scores = scores[keep_idx]
            if scores.dim() == 0:
                final_output.append(torch.empty(0, 5))
                continue
            keep_idx = nms(boxes_, scores, self.nms_iou_threshold)
            scores = scores[keep_idx].view(-1, 1)
            boxes_ = boxes_[keep_idx].view(-1, 4)
            output = torch.cat((boxes_, scores), dim=-1)
            final_output.append(output)
        return final_output

    def validate_detections(self, boxes: typing.List[np.ndarray]):
        for box in boxes:
            assert np.all(box[:, 4] <= 1) and np.all(box[:, 4] >= 0),\
                f"Confidence values not valid: {box}"
