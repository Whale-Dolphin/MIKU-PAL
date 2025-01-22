import os
import time

import click
import cv2
from einops import rearrange
import numpy as np
import torch
from torch.multiprocessing import Pool, set_start_method
import torchvision
from tqdm import tqdm

from models.dsfd.detect import DSFDDetector
from utils import sanitize_filenames_in_directory


def init_worker(args):
    """
    Initializes the detector for each process.
    """
    global Detector
    
    worker_id = torch.multiprocessing.current_process()._identity[0] - 1
    n_gpus = torch.cuda.device_count()
    gpu_id = worker_id % n_gpus
    
    device = torch.device(f'cuda:{gpu_id}' if torch.cuda.is_available() else 'cpu')
    args['device'] = device
    
    Detector = DSFDDetector(
        confidence_threshold=args['confidence_threshold'],
        nms_iou_threshold=args['nms_iou_threshold'],
        max_resolution=args['max_resolution'],
        device=device,
        fp16_inference=args['fp16_inference'],
        clip_boxes=args['clip_boxes'],
        model_path=args['model_path']
    )


def process_single_video(args):
    """
    Processes a single video, detects faces, and saves the results to files.
    """
    video_path, output_dir, data_name = args

    video_name = os.path.basename(video_path).split('.')[0]
    video_output_dir = os.path.join(output_dir, data_name, video_name)
    print(f"Video output directory: {video_output_dir}")
    os.makedirs(video_output_dir, exist_ok=True)
    
    video_reader = torchvision.io.VideoReader(video_path)

    print(f"Processing {video_path}")

    for frame_idx, frame in enumerate(video_reader):
        data = rearrange(frame['data'], 'c h w -> h w c').unsqueeze(0)
        bboxes = Detector.detect(data)[0][:,:4]

        pts = frame['pts']
        if len(bboxes) > 0:
            output_path = os.path.join(video_output_dir, f"{pts}_{len(bboxes)}.pt")
            torch.save(bboxes, output_path)


def process_videos(video_paths, output_dir, num_gpus, data_name, args):
    """
    Processes multiple videos in parallel.
    """
    with Pool(processes=num_gpus*10, initializer=init_worker, initargs=(args,)) as pool:
        for _ in tqdm(pool.imap(process_single_video, [(video_path, output_dir, data_name) for video_path in video_paths]), total=len(video_paths), desc='Processing Videos'):
            pass


@click.command()
@click.option('--input_dir', type=click.Path(exists=True), required=True)
@click.option('--data_name', type=str, required=True)
@click.option('--output_dir', type=click.Path(), default='outputs/face_detect')
@click.option('--confidence_threshold', type=float, default=.5)
@click.option('--nms_iou_threshold', type=float, default=.3)
@click.option('--max_resolution', type=int, default=1080)
@click.option('--shrink', type=float, default=1.0)
@click.option('--device', type=str, default='cuda')
@click.option('--fp16_inference', type=bool, default=True)
@click.option('--clip_boxes', type=bool, default=True)
@click.option('--model_path', type=str, default='checkpoints/DSFD.pth')
def main(input_dir, data_name, output_dir, confidence_threshold, nms_iou_threshold, max_resolution, shrink, device, fp16_inference, clip_boxes, model_path):
    set_start_method('spawn')
    args = click.get_current_context().params
    sanitize_filenames_in_directory(input_dir)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    video_extensions = ('.mp4', '.avi', '.mov', '.mkv')
    video_path = [os.path.join(input_dir, f) for f in os.listdir(input_dir) if f.endswith(video_extensions)]
    num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 1
    start_time = time.time()
    process_videos(video_path, output_dir, num_gpus, data_name, args)
    end_time = time.time()
    print(f"Processing time: {end_time - start_time:.2f} seconds")

if __name__ == '__main__':
    main()