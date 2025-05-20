from tqdm import tqdm
import polars as pl
from functools import partial
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

import os
from dotenv import dotenv_values

from packages.lang_identify import owsm_detect_language_from_array

MODEL_ID = "espnet/owsm_v3.1_ebf"
HF_CACHE_DIR = dotenv_values(".env")["HF_CACHE_DIR"]
SPEECHBOX_DIR = f"{HF_CACHE_DIR}/2152"
TARGET_SAMPLING_RATE = 16000
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"  # no mps support yet

s2l = Speech2Language.from_pretrained(
    model_tag=MODEL_ID,
    device=DEVICE,
    nbest=3  # return nbest prediction and probability
)

# s2t = Speech2Text.from_pretrained(
#     model_tag=MODEL_ID,
#     device=DEVICE,
#     beam_size=5,
#     ctc_weight=0.0,
#     maxlenratio=0.0,
#     # below are default values which can be overwritten in __call__
#     lang_sym="<eng>",
#     task_sym="<asr>",
#     predict_time=False,
# )

def _get_timestamp(seconds):
    return str(timedelta(seconds=seconds))[2:]

def process_dhr(identify_language_fn):
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
                prediction = identify_language_fn(sample)['language_prediction']
                predictions.append(prediction)
                gt_transcripts.append(gt_transcript)
                # s2t_transcripts.append()
                genders.append(identifier.split("_")[2])
                backgrounds.append(identifier.split("_")[3])
                timestamps.append(f"{_get_timestamp(first_interval_start)}-{_get_timestamp(first_interval_end)}")
                break
    frame = pl.DataFrame({
        "timestamp": timestamps,
        "gender": genders, 
        "langid_prediction": predictions,
        "background": backgrounds,
        "gt_transcript": gt_transcripts
    })
    ipdb.set_trace()
    return frame

# def speechbox_analysis():

def main():
    s2l = Speech2Language.from_pretrained(
    model_tag=MODEL_ID,
    device=DEVICE,
    nbest=3  # return nbest prediction and probability
    )
    identify_language_owsm_partial = partial(owsm_detect_language_from_array, s2l)
    identify_language_owsm = lambda sample_dict: identify_language_owsm_partial(sample_dict['audio']['array'])
    process_dhr(identify_language_owsm)

if __name__ == '__main__':
    main()