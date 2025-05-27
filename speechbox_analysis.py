import matplotlib.pyplot as plt
import seaborn as sns
import click
from tqdm import tqdm
import polars as pl
from functools import partial
import loguru
import ipdb
from datetime import timedelta
import torch
import math
import numpy as np
import librosa
from pathlib import Path
from praatio import textgrid
from espnet2.bin.s2t_inference_language import Speech2Language
from espnet2.bin.s2t_inference import Speech2Text
from jiwer import cer
# from flowmason import 
from flowmason import conduct, SingletonStep, load_artifact_with_step_name, MapReduceStep, load_mr_artifact


import os
from dotenv import dotenv_values
from collections import OrderedDict

from packages.lang_identify import owsm_detect_language_from_array, owsm_transcribe_from_array

logger = loguru.logger

MODEL_ID = "espnet/owsm_v3.1_ebf"
HF_CACHE_DIR = dotenv_values(".env")["HF_CACHE_DIR"]
SCRATCH_SAVE_DIR = dotenv_values(".env")['SCRATCH_SAVE_DIR']
MACHINE = dotenv_values(".env")["MACHINE"]
if MACHINE == "local":
    SPEECHBOX_DIR = f"allsstar/2152"
else:
    SPEECHBOX_DIR = f"{HF_CACHE_DIR}/2152"
TARGET_SAMPLING_RATE = 16000
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"  # no mps support yet



def _get_timestamp(seconds):
    return str(timedelta(seconds=seconds))[2:]

def process_dhr(identify_language_fn, inference_column, **kwargs):
    predictions = []
    backgrounds = []
    timestamps = []
    genders = []
    s2t_transcripts = []
    gt_transcripts = []
    for dirname in tqdm(list(filter(lambda x: x.endswith("DHR"), os.listdir(SPEECHBOX_DIR)))):
        speechbox_subdir = f"{SPEECHBOX_DIR}/{dirname}"
        identifiers = set([Path(dir_fname).stem for dir_fname in os.listdir(speechbox_subdir)])
        for identifier in identifiers:
            tg = textgrid.openTextgrid(f"{speechbox_subdir}/{identifier}.TextGrid", False)
            entries = tg.getTier('utt').entries
            data, _ = librosa.load(f"{speechbox_subdir}/{identifier}.wav", sr=TARGET_SAMPLING_RATE, dtype=np.float64)
            for i in range(len(entries)):
                first_interval_start, first_interval_end = entries[i].start, entries[i].end
                label = entries[i].label
                slice = data[math.floor(first_interval_start * TARGET_SAMPLING_RATE): math.ceil(first_interval_end * TARGET_SAMPLING_RATE)]
                sample = {
                    "audio": {
                        "array": slice,
                        "sampling_rate": TARGET_SAMPLING_RATE
                    }
                }
                gt_transcript = label.lower()
                if gt_transcript == "":
                    continue
                prediction = identify_language_fn(sample)[inference_column]
                predictions.append(prediction)
                gt_transcripts.append(gt_transcript)
                # s2t_transcripts.append()
                genders.append(identifier.split("_")[2])
                backgrounds.append(identifier.split("_")[3])
                timestamps.append(f"{_get_timestamp(first_interval_start)}-{_get_timestamp(first_interval_end)}")
    frame = pl.DataFrame({
        "timestamp": timestamps,
        "gender": genders, 
        f"{inference_column}_prediction": predictions,
        "background": backgrounds,
        "gt_transcript": gt_transcripts
    })
    return frame

def compute_cer(transcript_frame: pl.DataFrame, **kwargs):
    cer_frame = transcript_frame.with_columns(
        pl.struct(['gt_transcript', 'transcription_prediction']).apply(
            lambda row: cer(row['gt_transcript'], row['transcription_prediction'].lower())
        ).alias('cer')
    )
    return cer_frame

# def speechbox_analysis():

@click.command()
def detect_language():
    s2l = Speech2Language.from_pretrained(
        model_tag=MODEL_ID,
        device=DEVICE,
        nbest=3  # return nbest prediction and probability
    )
    identify_language_owsm_partial = partial(owsm_detect_language_from_array, s2l)
    identify_language_owsm = lambda sample_dict: identify_language_owsm_partial(sample_dict['audio']['array'])
    process_dhr(identify_language_owsm, 'langid')
    pass

def step_visualize_cer(model_name, transcript_frame, **kwargs):
    plot = sns.boxplot(data=transcript_frame.select('background', 'cer'), x='background', y='cer')
    fig = plot.get_fig()
    plt.tight_layout()
    output_path = 'owsm_output.png'
    fig.save_fig(output_path)
    logger.info(f'saved to {output_path}')

@click.command()
def transcribe_audio():
    s2t = Speech2Text.from_pretrained(
        model_tag=MODEL_ID,
        device=DEVICE,
        ctc_weight=0.0,
        maxlenratio=0.0,
        # below are default values which can be overwritten in __call__
        lang_sym="<eng>",
        task_sym="<asr>",
        predict_time=False,
    )
    transcribe_partial = partial(owsm_transcribe_from_array, s2t)
    identify_language_owsm = lambda sample_dict: transcribe_partial(sample_dict['audio']['array'])

    process_dhr_partial = partial(process_dhr, identify_language_owsm)
    step_dict = OrderedDict()
    step_dict['step_dhr_inference'] = SingletonStep(
        process_dhr_partial,
        {
            'inference_column': 'transcription', 
            'version': '001'
        }
    )
    step_dict['step_compute_cer'] = SingletonStep(
        compute_cer, 
        {
            'transcript_frame': 'step_dhr_inference',
            'version': '001'
        }
    )
    step_dict['visualize_cer'] = SingletonStep(
        step_visualize_cer, 
        {
            'model_name': 'owsm',
            'results_frame': 'step_compute_cer', 
            'version': '001'
        }
    )
    metadata = conduct(os.path.join(SCRATCH_SAVE_DIR, "tokenization_cache"), step_dict, "unstandard_speech_transcribe_speechbox")
    frame = load_artifact_with_step_name(metadata, 'step_dhr_inference')

@click.group()
def main():
    pass

main.add_command(detect_language)
main.add_command(transcribe_audio)

if __name__ == '__main__':
    main()