import os
import time

import click
import cv2
import torch
from torch.multiprocessing import Pool, set_start_method
import torchvision
from tqdm import tqdm

from detect import DSFDDetector

def init_worker(args):
    """
    Initializes the detector for each process.
    """
    global Detector
    Detector = DSFDDetector(**vars(args))
    if args.use_cuda:
        Detector = Detector.cuda()


def process_videos(video_paths, output_dir, num_gpus, args):
    """
    Processes multiple videos in parallel.
    """
    with Pool(processes=num_gpus, initializer=init_worker, initargs=(args,)) as pool:
        for _ in tqdm(pool.imap(process_single_video, [(video_path, output_dir) for video_path in video_paths]), total=len(video_paths), desc='Processing Videos'):
            pass


@click.command()
@click.option('--input_dir', type=click.Path(exists=True), required=True)
@click.option('--output_dir', type=click.Path(), default='../fd_outputs')
@click.option('--confidence_threshold', type=float, default=.5)
@click.option('--nms_iou_threshold', type=float, default=.3)
@click.option('--max_resolution', type=int, default=1080)
@click.option('--shrink', type=float, default=1.0)
@click.option('--device', type=str, default='cuda')
@click.option('--fp16_inference', type=bool, default=True)
@click.option('--clip_boxes', type=bool, default=True)
@click.option('--model_path', type=str, default='../checkpoints/DSFD.pth')
def main(input_dir, output_dir, confidence_threshold, nms_iou_threshold, max_resolution, shrink, device, fp16_inference, clip_boxes, model_path):
    set_start_method('spawn')
    args = click.get_current_context().params
    # print(args)
    
    video_extensions = ('.mp4', '.avi', '.mov', '.mkv')
    video_path = [os.path.join(input_dir, f) for f in os.listdir(input_dir) if f.endswith(video_extensions)]
    num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 1
    start_time = time.time()
    process_video(video_path, output_dir, num_gpus, args)
    end_time = time.time()
    print(f"Processing time: {end_time - start_time:.2f} seconds")

if __name__ == '__main__':
    main()