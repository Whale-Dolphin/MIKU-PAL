# MIKU-PAL:An Automatic Multi-Modal Method for \\ Audio Paralinguistic and Affect Labeling

MIKU-PAL is an antomatic tool for emotion speech. MIKU-PAL stands for Multi-Modal Intelligence Kit for Understanding - Paralinguistic and Affect Labeling. Also can be understood as pal of Miku, which can help Miku understand human emotion.

## Introduction

MIKU-PAL have 5 stage in engineering,

- Stage 1: Turn video information to audio information. This stage will maintain the samplerates of audio.
- Stage 2: UVR. Use UVR5 to seperate the speech and background noise and protential bgm.
- Stage 3: ASR. Use whisper-large to do ASR and generate the timestamp of each sentence.
- Stage 4: ASD. This module inherits face detection. you can use either the S^3FD provided by TalkNet ASD or the DSFD provided by MIKU-PAL. the speaker segments with the highest confidence in each scene will eventually be saved in outputfinal_video_list.list and outputfinal_wav_list.list.
- Stage 5: VLM FER. Use gemini-experiment-1206 to analyse the face emotion and save the final audio with rename it to <Emotion><Text>.

## Usage

```
    git clone https://github.com/Whale-Dolphin/MIKU-PAL.git
    cd MIKU-PAL
```

```
    conda create -n miku-pal python=3.11
    conda activate miku-pal
```

Then put the taget folder to inputdir.list with one line one path. Then run:
```
    bash parallel.sh
```
The kit will automatically analyse GPU number and use all GPUs.

(P.S. The default params will just run out of the 24G gpu memories. Tested on 4090 Ubuntu22.04)