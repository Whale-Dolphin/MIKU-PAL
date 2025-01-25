import os
import time
import cv2

import torchvision
from einops import rearrange
import numpy as np
import torch

from models.dsfd.detect import DSFDDetector
from utils import sanitize_filenames_in_directory


class FaceDetect:
    def __init__(
        self, 
        confidence_threshold = 0.5, 
        nms_iou_threshold = 0.3, 
        max_resolution = 640, 
        device = 'cuda', 
        fp16_inference = True, 
        clip_boxes = True, 
        model_path = 'checkpoints/dsfd.pth', 
        batch_size = 32
        ):
        self.detector = DSFDDetector(
            confidence_threshold=confidence_threshold,
            nms_iou_threshold=nms_iou_threshold,
            max_resolution=max_resolution,
            device=device,
            fp16_inference=fp16_inference,
            clip_boxes=clip_boxes,
            model_path=model_path
        )
        self.batch_size = batch_size

    def run(self, video_tensor):
        all_detections = []
        timestamp_buffer = []
        H, W, C = video_tensor[0].shape
        frame_buffer = torch.empty(0, H, W, C, dtype=torch.uint8)
        for frame_idx, frame in enumerate(video_tensor):
            frame_buffer = torch.cat((frame_buffer, frame.unsqueeze(0)), dim=0)
            timestamp_buffer.append(frame_idx)
            
            if len(frame_buffer) == self.batch_size:
                detected_batch = self.detector.detect(frame_buffer)
                for num, i in enumerate(detected_batch):
                    all_detections.append([])
                    for j in i:
                        all_detections[-1].append({
                            'frame': timestamp_buffer[num],
                            'bbox': (j[:4]).tolist(),
                            'conf': (j[4]).tolist()
                        })
                
                H, W, C = frame_buffer.shape[1:]
                frame_buffer = torch.empty(0, H, W, C, dtype=torch.uint8)
                timestamp_buffer = []

        return all_detections


def main(input_dir, data_name, output_dir, confidence_threshold, nms_iou_threshold, max_resolution, shrink, device, fp16_inference, clip_boxes, model_path):
    sanitize_filenames_in_directory(input_dir)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    video_extensions = ('.mp4', '.avi', '.mov', '.mkv')
    video_paths = [os.path.join(input_dir, f) for f in os.listdir(
        input_dir) if f.endswith(video_extensions)]

    start_time = time.time()
    face_detect = FaceDetect(confidence_threshold, nms_iou_threshold,
                             max_resolution, device, fp16_inference, clip_boxes, model_path)
    face_detect.process_videos(video_paths, output_dir, data_name)
    end_time = time.time()
    print(f"Processing time: {end_time - start_time:.2f} seconds")


if __name__ == '__main__':
    main()
