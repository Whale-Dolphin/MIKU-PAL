import sys
import os

sys.path.append('models/')

from loguru import logger

from models.fish_data_engine.tasks.uvr import UVR
from models.fish_data_engine.tasks.align import WhisperAlignTask
from models.fish_data_engine.tasks.asr import ASR


logger.add('pipeline.log', format="{time} {level} {message}", level="DEBUG")

class Pipeline:
    def __init__(self, input_dir, output_dir):
        self.input_dir = input_dir
        self.output_dir = output_dir

    def run(self):
        self.run_uvr()
        self.run_asr()

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

    def run_face_detect()
