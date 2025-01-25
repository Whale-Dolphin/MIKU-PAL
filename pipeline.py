import sys
import os
import glob
import subprocess
from concurrent.futures import ThreadPoolExecutor

import av
import click
from loguru import logger
from torchvision.io import read_video
from tqdm import tqdm

from models.fish_data_engine.tasks.uvr import UVR
from models.fish_data_engine.tasks.align import WhisperAlignTask
from models.fish_data_engine.tasks.asr import ASR
from models.dsfd.inference import FaceDetect
from models.TalkNet_ASD.inference import crop_video
from gemini import run_gemini


logger.add('pipeline.log', format="{time} {level} {message}", level="DEBUG")

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


class Pipeline:
    def __init__(self, input_dir, output_dir):
        self.input_dir = input_dir
        self.output_dir = output_dir

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

        logger.debug(f'Running UVR with input={input_dir}, output={output_dir}')

        uvr = UVR(
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

        logger.debug(f'Running ASR with input={input_dir}, output={output_dir}')

        asr = ASR(
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

    def run_face_detect(
        self,
        frames,
        confidence_threshold=0.5,
        nms_iou_threshold=0.3,
        max_resolution=640,
        shrink=1,
        device='cuda',
        fp16_inference=True,
        clip_boxes=True,
        model_path='checkpoints/dsfd.pth',
        data_name='face_detect',
    ):  
        # logger.debug(f'Running Face Detect with input={input_dir}, output={output_dir}')

        self.fd = FaceDetect(
            confidence_threshold=confidence_threshold,
            nms_iou_threshold=nms_iou_threshold,
            max_resolution=max_resolution,
            device=device,
            fp16_inference=fp16_inference,
            clip_boxes=clip_boxes,
            model_path=model_path
        )

    def run_asd(
        self,
        input_path,
        asr_output_path,
        output_path,
        model_path = 'checkpoints/pretrain_TalkSet.model',
        numFailedDet = 10,
        minTrack = 10,
        minFaceSize = 1,

    ):
        logger.debug(f'Running ASD with input={input_dir}, output={output_dir}')

        fps = get_video_fps(input_path)
        with open(asr_output_path, 'r') as f:
            data = json.laod(f)

        for i, scene in enumerate(data):
            allTracks, vidTracks = [], []
            scene_start = scene['start']
            scene_end = scene['end']

            frames, audios, info = read_video(
                video_path=input_path,
                start_pts=scene_start,
                end_pts=scene_end,
                pts_unit='sec')

            info['fps'] = fps

            faces = self.fd.run(frames)

            allTracks.extend(track_shot(numFailedDet, minTrack, minFaceSize, faces))

            cropPath = f'{output_path}/croped/{i}'
            os.makedir(cropPath, exist_ok=True)
            for ii, track in tqdm(enumerate(allTracks), total=len(allTracks)):
                vidTracks.append(crop_video(frames, audios, track, info,
                                            f'{cropPath}/{ii:05d}'))

            files = glob.glob("%s/*.avi" % cropPath)

            files.sort()
            scores = evaluate_network(files, model_path, cropPath)

            for tidx, track in enumerate(vidTracks):
                score = scores[tidx]
                for fidx, frame in enumerate(track['track']['frame'].tolist()):
                    s = np.mean(score[max(fidx - 2, 0): min(fidx + 3, len(score) - 1)])
                    faces[frame].append({'track': tidx, 'score': float(s), 's': track['proc_track']['s']
                                    [fidx], 'x': track['proc_track']['x'][fidx], 'y': track['proc_track']['y'][fidx]})

            track_scores = {}
            for frame_faces in faces:
                for face_info in frame_faces:
                    t_id = face_info['track']
                    track_scores.setdefault(t_id, []).append(face_info['score'])

            best_track, best_score = None, float('-inf')
            for t_id, scores_list in track_scores.items():
                avg_score = sum(scores_list) / len(scores_list)
                if avg_score > best_score:
                    best_track, best_score = t_id, avg_score

            best_track_mp4 = f"{cropPath}/{best_track}.mp4"
            best_track_wav = f"{cropPath}/{best_track}.wav"
            final_mp4 = f"{output_path}/{i}.mp4"

            cmd = [
                "ffmpeg", "-y",
                "-i", best_track_mp4,
                "-i", best_track_wav,
                "-c:v", "copy",
                "-c:a", "aac",
                final_mp4
            ]
            subprocess.run(cmd, check=True)

        return

    def run(
        self,
        input_dir,
        output_dir
    ):
        video_files = []
        for pattern in ("*.mp4", "*.avi", "*.mkv", "*.mov"):
            video_files.extend(glob.glob(os.path.join(input_dir, pattern)))
        print(f"Found video files: {video_files}")
        
        os.makedirs(os.path.join(output_dir, "raw_audio"), exist_ok=True)

        def extract_audio(video_file):
            base_name = os.path.splitext(os.path.basename(video_file))[0]
            output_wav = os.path.join(output_dir, "raw_audio", f"{base_name}.wav")
            cmd = [
            "ffmpeg",
            "-y",
            "-i", video_file,
            "-vn",
            "-acodec", "pcm_s16le",
            "-ar", "44100",
            "-ac", "2",
            "-threads", "8",
            output_wav
            ]
            subprocess.run(cmd, check=True)

        with ThreadPoolExecutor(max_workers=32) as executor:
            executor.map(extract_audio, video_files)

        os.makedir(os.path.join(output_dir, "uvr"), exist_ok=True)
        self.run_uvr(f"{output_dir}/raw_audio", f"{output_dir}/uvr")
        os.makedir(os.path.join(output_dir, "asr"), exist_ok=True)
        self.run_asr(f"{output_dir}/uvr", f"{output_dir}/asr")
        os.makedir(os.path.join(output_dir, "asd"), exist_ok=True)
        for video_file in video_files:
            best_track, best_score = self.run_asd(f"{video_file}", f"{output_dir}/asr/{video_file.split('.')[:-1]}/asr.json", f"{output_dir}/asd")
        
        # video_paths = []
        # for mp4_file in glob.glob(os.path.join(output_dir, "asd", "*.mp4")):
        #     video_path.append(mp4_file)

        # for video in video_paths:
        #     gemini_output = run_gemini(video)
        #     emotion_info = find_words_in_sentence(
        #         text_normalize[gemini_output.split('.')[0]], EMOTION_CATEGORIES)

        #     with open(os.path.join(output_dir, "emotion_audio.json"), "a") as f:
        #         f.write(f'{{"video": "{video}", "gemini_output": "{gemini_output}", "emotion_info": {emotion_info}}}\n')


@click.command()
@click.option("--input-dir", "-i", required=True, help="Input directory containing video files")
@click.option("--output-dir", "-o", required=True, help="Output directory")
def main(input_dir, output_dir):
    pipeline = Pipeline(input_dir, output_dir)
    pipeline.run(input_dir, output_dir)

if _main_ == "_main_":
    main()