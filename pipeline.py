from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
import glob
import json
import os
import shutil
import subprocess
import sys

import av
import click
from loguru import logger
import numpy as np
import soundfile as sf
from torchvision.io import read_video
from tqdm import tqdm

from models.fish_data_engine.tasks.uvr import UVR
from models.fish_data_engine.tasks.align import WhisperAlignTask
from models.fish_data_engine.tasks.asr import ASR
from models.dsfd.inference import FaceDetect
from models.TalkNet_ASD.inference import crop_video
from models.TalkNet_ASD.inference import track_shot
from models.TalkNet_ASD.inference import evaluate_network
from gemini import run_gemini


os.makedirs('logs', exist_ok=True)
log_filename = f'logs/pipeline_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'
logger.add(log_filename, format="{time} {level} {message}", level="DEBUG")

EMOTION_CATEGORIES = [
    "angry",
    "happy",
    "sad",
    "neutral",
    "frustrated",
    "excited",
    "fearful",
    "surprised",
    "disgusted",
]


def get_video_fps(video_path):
    try:
        with av.open(video_path) as container:
            video_stream = container.streams.video[0]
            fps = video_stream.average_rate
            return fps
    except Exception as e:
        print(f"Error reading video: {e}")
        return None


def text_normalize(text):
    text = text.replace(" ", "")
    text = text.replace("\n", "")
    text = text.replace("\r", "")
    text = text.replace("\t", "")
    text = text.lower()
    return text


def find_words_in_sentence(sentence, words):
    found_words = []
    for word in words:
        if word in sentence:
            found_words.append(word)
    return found_words


class ProcessorFactory:
    @staticmethod
    def create_uvr(**kwargs):
        return UVR(**kwargs)

    @staticmethod
    def create_asr(**kwargs):
        return ASR(**kwargs)
 

class Pipeline:
    def __init__(self, input_dir, output_dir):
        self.input_dir = input_dir
        self.output_dir = output_dir
        confidence_threshold = 0.5
        nms_iou_threshold = 0.3
        max_resolution = 640
        shrink = 1
        device = 'cuda'
        fp16_inference = True
        clip_boxes = True
        model_path = 'checkpoints/dsfd.pth'
        data_name = 'face_detect'
        print(type(model_path))
        self.fd = FaceDetect(
            confidence_threshold=confidence_threshold,
            nms_iou_threshold=nms_iou_threshold,
            max_resolution=max_resolution,
            device=device,
            fp16_inference=fp16_inference,
            clip_boxes=clip_boxes,
            model_path=model_path
        )
        self.factory = ProcessorFactory()

    def run_uvr(
        self,
        input_dir=None,
        output_dir=None,
        demix=True,
        demix_model="mdx23c_vip",
        demix_segment_size=None,
        demix_overlap=0.25,
        demix_batch_size=4,
        log_interval=0,
        deecho_deverb=True,
        vocals_only=True,
        use_quality_model=False,
        save_format='wav',
    ):
        if input_dir is None:
            input_dir = os.path.join(self.input_dir, 'uvr')
        if output_dir is None:
            output_dir = os.path.join(self.output_dir, 'uvr')

        logger.debug(
            f'Running UVR with input={input_dir}, output={output_dir}')

        uvr = self.factory.create_uvr(
            input_dir=input_dir,
            output_dir=output_dir,
            demix=demix,
            demix_model=demix_model,
            demix_segment_size=demix_segment_size,
            demix_overlap=demix_overlap,
            demix_batch_size=demix_batch_size,
            log_interval=log_interval,
            deecho_deverb=deecho_deverb,
            vocals_only=vocals_only,
            use_quality_model=use_quality_model,
            save_format=save_format,
        )
        uvr.run()

    def run_asr(
        self,
        input_dir=None,
        output_dir=None,
        language='en',
        punctuation=True,
        align=True,
        diarize=True,
        drop_sentence_threshold=0.7,
        save_format='wav',
        filter_intonation=False,
        additional_save='None'
    ):
        if input_dir is None:
            input_dir = os.path.join(self.input_dir, 'asr')
        if output_dir is None:
            output_dir = os.path.join(self.output_dir, 'asr')

        logger.debug(
            f'Running ASR with input={input_dir}, output={output_dir}')

        asr = self.factory.create_asr(
            input_dir=input_dir,
            output_dir=output_dir,
            language=language,
            punctuation=punctuation,
            align=align,
            diarize=diarize,
            drop_sentence_threshold=drop_sentence_threshold,
            save_format=save_format,
            filter_intonation=filter_intonation,
            additional_save=additional_save,
        )
        asr.run()

    def run_asd(
        self,
        input_path,
        asr_output_path,
        output_path,
        audio_path,
        model_path='checkpoints/pretrain_TalkSet.model',
        numFailedDet=10,
        minTrack=10,
        minFaceSize=1,

    ):
        logger.debug(
            f'Running ASD with input={input_path}, asr_output={asr_output_path}, [[output={output_path}')

        fps = get_video_fps(input_path)
        with open(asr_output_path, 'r') as f:
            data = json.load(f)

        logger.debug(data)

        final_mp4s, best_track_wavs = [], []

        for i, scene in enumerate(data):
            allTracks, vidTracks = [], []
            scene_start = scene['start']
            scene_end = scene['end']

            frames, _, info = read_video(
                input_path,
                start_pts=scene_start,
                end_pts=scene_end,
                pts_unit='sec')

            start_frame = int(scene_start * info['audio_fps'])
            end_frame = int(scene_end * info['audio_fps'])
            audios, _ = sf.read(
                audio_path,
                start=start_frame,
                frames=end_frame - start_frame,
            )

            logger.debug(frames.shape)

            info['fps'] = fps

            faces = self.fd.run(frames)
            if not faces:
                continue


            allTracks.extend(track_shot(
                numFailedDet, minTrack, minFaceSize, faces))

            cropPath = f'{output_path}/croped/{i}'
            os.makedirs(cropPath, exist_ok=True)
            for ii, track in tqdm(enumerate(allTracks), total=len(allTracks)):
                vidTracks.append(crop_video(frames, audios, track, info,
                                            f'{cropPath}/{ii:05d}'))

            files = glob.glob("%s/*.avi" % cropPath)

            files.sort()
            scores = evaluate_network(files, model_path, cropPath)

            for tidx, track in enumerate(vidTracks):
                score = scores[tidx]
                for fidx, frame in enumerate(track['track']['frame'].tolist()):
                    s = np.mean(
                        score[max(fidx - 2, 0): min(fidx + 3, len(score) - 1)])
                    faces[frame].append({'track': tidx, 'score': float(s), 's': track['proc_track']['s']
                                         [fidx], 'x': track['proc_track']['x'][fidx], 'y': track['proc_track']['y'][fidx]})

            track_scores = {}
            for frame_faces in faces:
                for face_info in frame_faces:
                    t_id = face_info['track']
                    track_scores.setdefault(
                        t_id, []).append(face_info['score'])

            best_track, best_score = None, float('-inf')
            for t_id, scores_list in track_scores.items():
                avg_score = sum(scores_list) / len(scores_list)
                if avg_score > best_score:
                    best_track, best_score = t_id, avg_score

            best_track_mp4 = f"{cropPath}/{best_track:05d}.mp4"
            best_track_wav = f"{cropPath}/{best_track:05d}.wav"
            final_mp4 = f"{output_path}/{i}.mp4"

            final_mp4s.append(final_mp4)
            best_track_wavs.append(best_track_wav)

            cmd = [
                "ffmpeg", "-y",
                "-i", best_track_mp4,
                "-i", best_track_wav,
                "-c:v", "copy",
                "-c:a", "aac",
                final_mp4
            ]
            subprocess.run(cmd, check=True)

        return final_mp4s, best_track_wavs

    def run(
        self,
        input_dir,
        output_dir,
        stage=0,
        stop_stage=3,
    ):
        logger.info(
            f"Running pipeline with input_dir={input_dir}, output_dir={output_dir}, stage={stage}, stop_stage={stop_stage}")
        video_files = []
        for pattern in ("*.mp4", "*.avi", "*.mkv", "*.mov"):
            video_files.extend(glob.glob(os.path.join(input_dir, pattern)))
        logger.info(f"Found video files: {video_files}")
        if stage == 0 and stop_stage >= 0:
            logger.info('Stage 0: Extract raw audio files')

            os.makedirs(os.path.join(output_dir, "raw_audio"), exist_ok=True)

            def extract_audio(video_file):
                base_name = os.path.splitext(os.path.basename(video_file))[0]
                output_wav = os.path.join(
                    output_dir, "raw_audio", f"{base_name}.wav")
                cmd = [
                    "ffmpeg",
                    "-y",
                    "-i", video_file,
                    "-vn",
                    "-acodec", "pcm_s16le",
                    output_wav
                ]
                subprocess.run(cmd, check=True)

            with ThreadPoolExecutor(max_workers=32) as executor:
                executor.map(extract_audio, video_files)

            stage += 1

        os.makedirs(os.path.join(output_dir, "uvr"), exist_ok=True)
        raw_audio_dir = os.path.abspath(os.path.join(output_dir, "raw_audio"))
        uvr_output_dir = os.path.abspath(os.path.join(output_dir, "uvr"))
        if stage == 1 and stop_stage >= 1:
            logger.info('Stage 1: Run UVR')

            self.run_uvr(raw_audio_dir, uvr_output_dir)
            shutil.rmtree(raw_audio_dir, ignore_errors=True)
            stage += 1

        os.makedirs(os.path.join(output_dir, "asr"), exist_ok=True)
        asr_output_dir = os.path.abspath(os.path.join(output_dir, "asr"))
        if stage == 2 and stop_stage >= 2:
            logger.info('Stage 2: Run ASR')
            logger.debug(f"ASR inputdir {uvr_output_dir}, ASR outputdir {asr_output_dir}")
            self.run_asr(uvr_output_dir, asr_output_dir)
            stage += 1

        if stage == 3 and stop_stage >= 3:
            final_mp4ss, best_track_wavss = [], []
            logger.info('Stage 3: Run Face Detect and ASD')
            os.makedirs(os.path.join(output_dir, "asd"), exist_ok=True)
            for video_file in video_files:
                video_name = os.path.splitext(os.path.basename(video_file))[0]
                final_mp4s, best_track_wavs = self.run_asd(
                    f"{video_file}", f"{output_dir}/asr/{video_name}/asr.json", f"{output_dir}/asd/{video_name}", f"{output_dir}/uvr/{video_name}.wav")
                for i, j in zip(final_mp4s, best_track_wavs):
                    final_mp4ss.append(i)
                    best_track_wavss.append(j)
            with open(f'{output_dir}final_video_list.list', 'w') as f1:
                f1.write('\n'.join(final_mp4ss))
            with open(f'{output_dir}final_wav_list.list', 'w') as f2:
                f2.write('\n'.join(best_track_wavss))
            stage += 1

        if stage == 4 and stop_stage >= 4:
            logger.info('Stage 4: Run Gemini')
            video_paths = []
            for mp4_file in glob.glob(os.path.join(output_dir, "asd", "*.mp4")):
                video_path.append(mp4_file)

            for video in video_paths:
                gemini_output = run_gemini(video)
                emotion_info = find_words_in_sentence(
                    text_normalize[gemini_output.split('.')[0]], EMOTION_CATEGORIES)

                with open(os.path.join(output_dir, "emotion_audio.json"), "a") as f:
                    f.write(
                        f'{{"video": "{video}", "gemini_output": "{gemini_output}", "emotion_info": {emotion_info}}}\n')


@click.command()
@click.option("--input-dir", "-i", required=True, help="Input directory containing video files")
@click.option("--output-dir", "-o", required=True, help="Output directory")
@click.option("--stage", default=0, help="Start stage")
@click.option("--stop-stage", default=3, help="Stop stage")
def main(input_dir, output_dir, stage, stop_stage):
    pipeline = Pipeline(input_dir, output_dir)
    pipeline.run(input_dir, output_dir, stage, stop_stage)


if __name__ == "__main__":
    main()
