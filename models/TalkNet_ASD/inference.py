import os
import math

import numpy as np
import cv2
import soundfile as sf
import torch
from scipy import signal
from scipy.io import wavfile
from scipy.interpolate import interp1d
from torchvision import transforms
from einops import rearrange
import python_speech_features
from tqdm import tqdm

from models.TalkNet_ASD.talknet import talkNet

def bb_intersection_over_union(boxA, boxB, evalCol=False):
	# CPU: IOU Function to calculate overlap between two image
	xA = max(boxA[0], boxB[0])
	yA = max(boxA[1], boxB[1])
	xB = min(boxA[2], boxB[2])
	yB = min(boxA[3], boxB[3])
	interArea = max(0, xB - xA) * max(0, yB - yA)
	boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
	boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
	if evalCol == True:
		iou = interArea / float(boxAArea)
	else:
		iou = interArea / float(boxAArea + boxBArea - interArea)
	return iou


def track_shot(numFailedDet, minTrack, minFaceSize, sceneFaces):
	# CPU: Face tracking
	iouThres = 0.5     # Minimum IOU between consecutive face detections
	tracks = []
	while True:
		track = []
		for frameFaces in sceneFaces:
			for face in frameFaces:
				if track == []:
					track.append(face)
					frameFaces.remove(face)
				elif face['frame'] - track[-1]['frame'] <= numFailedDet:
					iou = bb_intersection_over_union(face['bbox'], track[-1]['bbox'])
					if iou > iouThres:
						track.append(face)
						frameFaces.remove(face)
						continue
				else:
					break
		if track == []:
			break
		elif len(track) > minTrack:
			frameNum = np.array([f['frame'] for f in track])
			bboxes = np.array([np.array(f['bbox']) for f in track])
			frameI = np.arange(frameNum[0], frameNum[-1]+1)
			bboxesI = []
			for ij in range(0, 4):
				interpfn = interp1d(frameNum, bboxes[:, ij])
				bboxesI.append(interpfn(frameI))
			bboxesI = np.stack(bboxesI, axis=1)
			if max(np.mean(bboxesI[:, 2]-bboxesI[:, 0]), np.mean(bboxesI[:, 3]-bboxesI[:, 1])) > minFaceSize:
				tracks.append({'frame': frameI, 'bbox': bboxesI})
	return tracks


def crop_video(video_tensor, audio_tensor, track, info, crop_file):
    """Crops a video and audio tensor based on tracking data and saves the result to local.

    Args:
        video_tensor: A tensor of shape (T, C, H, W) representing the video.
        audio_tensor: A tensor of shape (S,) representing the audio waveform. S is the number of samples.
        track: A dictionary containing tracking information, including 'bbox' (bounding boxes) and 'frame' (frame indices).
        crop_file: The base filename for saving the cropped video and audio.

    Returns:
        A dictionary containing the original `track`, the processed detection data `dets`, and the path of the cropped video and audio.
        Returns None if track or video_tensor is invalid.
    """

    if video_tensor is None or audio_tensor is None or track is None or 'bbox' not in track or 'frame' not in track:
        return None

    audio_sample_rate = info['audio_fps']
    video_fps = info['video_fps']

    dets = {'x': [], 'y': [], 's': []}
    for det in track['bbox']:
        dets['s'].append(max((det[3] - det[1]), (det[2] - det[0])) / 2)
        dets['y'].append((det[1] + det[3]) / 2)
        dets['x'].append((det[0] + det[2]) / 2)

    dets['s'] = signal.medfilt(dets['s'], kernel_size=13)
    dets['x'] = signal.medfilt(dets['x'], kernel_size=13)
    dets['y'] = signal.medfilt(dets['y'], kernel_size=13)

    cropped_frames = []
    for fidx, frame_index in enumerate(track['frame']):
        if frame_index >= video_tensor.shape[0]:
            continue
        frame = video_tensor[frame_index].numpy() * 255
        frame = frame.astype(np.uint8)
        bs = dets['s'][fidx]
        bsi = int(bs * (1 + 2 * 1.0))

        # Pad the frame
        frame = np.pad(frame, ((bsi, bsi), (bsi, bsi), (0, 0)),
                       'constant', constant_values=(110, 110))

        my = dets['y'][fidx] + bsi
        mx = dets['x'][fidx] + bsi
        try:
            face = frame[int(my - bs):int(my + bs * (1 + 2 * 1.0)),
                         int(mx - bs * (1 + 1.0)):int(mx + bs * (1 + 1.0))]
        except:
            continue
        face = cv2.resize(face, (224, 224))

        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[
                                 0.229, 0.224, 0.225])
        ])
        face_tensor = transform(face)
        cropped_frames.append(face_tensor)

    if len(cropped_frames) == 0:
        return None

    cropped_video_tensor = torch.stack(cropped_frames)

    video_path = crop_file + ".avi"
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(video_path, fourcc, video_fps, (224, 224))
    for frame in cropped_video_tensor:
        frame = (frame.numpy() * 255).astype(np.uint8)
        out.write(frame)
    out.release()

    audio_start_frame = track['frame'][0]
    audio_end_frame = track['frame'][-1]
    audio_start_sample = int(audio_start_frame / video_fps *
                             audio_sample_rate)
    audio_end_sample = int((audio_end_frame + 1) /
                           video_fps * audio_sample_rate)
    cropped_audio = audio_tensor[:, audio_start_sample:audio_end_sample]
    audio_path = crop_file + ".wav"
    cropped_audio = rearrange(cropped_audio, 'c s -> s c')
    sf.write(audio_path, cropped_audio.numpy(), audio_sample_rate, format='WAV')

    return {'track': track, 'proc_track': dets, 'video_path': video_path, 'audio_path': audio_path}


def evaluate_network(files, pretrainModel, pycropPath):
	s = talkNet()
	s.loadParameters(pretrainModel)
	s.eval()
	allScores = []
	durationSet = {1, 1, 1, 2, 2, 2, 3, 3, 4, 5, 6}
	for file in tqdm(files, total=len(files)):
		fileName = os.path.splitext(file.split('/')[-1])[0]
		_, audio = wavfile.read(os.path.join(pycropPath, fileName + '.wav'))
		audioFeature = python_speech_features.mfcc(
			audio, 16000, numcep=13, winlen=0.025, winstep=0.010)
		video = cv2.VideoCapture(os.path.join(pycropPath, fileName + '.avi'))
		videoFeature = []
		while video.isOpened():
			ret, frames = video.read()
			if ret == True:
				face = cv2.cvtColor(frames, cv2.COLOR_BGR2GRAY)
				face = cv2.resize(face, (224, 224))
				face = face[int(112-(112/2)):int(112+(112/2)),
                                    int(112-(112/2)):int(112+(112/2))]
				videoFeature.append(face)
			else:
				break
		video.release()
		videoFeature = np.array(videoFeature)
		length = min((audioFeature.shape[0] - audioFeature.shape[0] %
		             4) / 100, videoFeature.shape[0] / 25)
		audioFeature = audioFeature[:int(round(length * 100)), :]
		videoFeature = videoFeature[:int(round(length * 25)), :, :]
		allScore = []  # Evaluation use TalkNet
		for duration in durationSet:
			batchSize = int(math.ceil(length / duration))
			scores = []
			with torch.no_grad():
				for i in range(batchSize):
					inputA = torch.FloatTensor(
						audioFeature[i * duration * 100:(i+1) * duration * 100, :]).unsqueeze(0).cuda()
					inputV = torch.FloatTensor(
						videoFeature[i * duration * 25: (i+1) * duration * 25, :, :]).unsqueeze(0).cuda()
					embedA = s.model.forward_audio_frontend(inputA)
					embedV = s.model.forward_visual_frontend(inputV)
					embedA, embedV = s.model.forward_cross_attention(embedA, embedV)
					out = s.model.forward_audio_visual_backend(embedA, embedV)
					score = s.lossAV.forward(out, labels=None)
					scores.extend(score)
			allScore.append(scores)
		allScore = np.round(
			(np.mean(np.array(allScore), axis=0)), 1).astype(float)
		allScores.append(allScore)
	return allScores
