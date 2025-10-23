# import polars as pl
import pandas as pd
import ipdb
import numpy as np
from string import punctuation
import torch
import os
from tqdm import tqdm
import soundfile as sf
from jiwer import wer as jiwer_wer
import math
from collections import Counter
import librosa
from praatio import textgrid 
from dotenv import dotenv_values
from functools import partial
import click

from packages.lang_identify import identify_language_speechbrain, owsm_detect_language_from_array
from packages.mms import mms_transcribe_from_array, delete_model_mms
from packages.whisper import whisper_transcribe_from_array
from packages.qwen import qwen_transcribe_from_array, delete_model_qwen
# from packages.canary import canary_transcribe_from_array
from packages.owsm import owsm_transcribe_from_array, delete_model_owsm

SCRATCH_DIR = dotenv_values(".env")["SCRATCH_DIR"]
HF_CACHE_DIR = dotenv_values(".env")["HF_CACHE_DIR"]

SPICE_DIRNAME = f"spice"
TARGET_SAMPLING_RATE= 16000

def remove_punctuation(text):
    return "".join([c for c in text if c not in punctuation])

def wer(prediction, ground_truth):
    return jiwer_wer(
        remove_punctuation(ground_truth.lower()), 
        remove_punctuation(prediction.lower()))

def transcribe_valid_snippets(model_name, dtype, participant_id):
    delete_model_fn = None
    if model_name.startswith('whisper'): 
        transcribe_fn = partial(whisper_transcribe_from_array, model=model_name[len('whisper-'):], language="en")
    elif model_name.startswith('mms'):
        transcribe_fn = partial(mms_transcribe_from_array, language="eng")
        delete_model_fn = delete_model_mms
    elif model_name.startswith('qwen'):
        transcribe_fn = partial(qwen_transcribe_from_array)
        delete_model_fn = delete_model_qwen
    # elif model_name.startswith('canary'):
    #     transcribe_fn = partial(canary_transcribe_from_array) 
    elif model_name.startswith('owsm'):
        transcribe_fn = partial(owsm_transcribe_from_array) 
        delete_model_fn = delete_model_owsm
    else:
        raise ValueError(f"Unknown model name {model_name}")

    participant_full_wav_file = get_participant_wav_file(participant_id)
    annotated_tg = get_annotated_textgrid(participant_id)
    entries = textgrid.openTextgrid(annotated_tg, includeEmptyIntervals=True).getTier('is-valid').entries
    transcript_entries = textgrid.openTextgrid(annotated_tg, includeEmptyIntervals=True).getTier('utterance').entries
    # only keep entries where the label is 'Y'
    data, _ = librosa.load(participant_full_wav_file, sr=TARGET_SAMPLING_RATE, dtype=dtype)
    predictions = []
    transcripts = []
    for i in range(len(entries)):
        if entries[i][2].strip() != 'Y':
            continue
        transcript = transcript_entries[i][2].strip()
        first_interval_start, first_interval_end = entries[i].start, entries[i].end
        slice = data[math.floor(first_interval_start * TARGET_SAMPLING_RATE): math.ceil(first_interval_end * TARGET_SAMPLING_RATE)]
        # print(samplerate)
        prediction = transcribe_fn(slice)
        predictions.append(prediction)
        transcripts.append(transcript)
    if delete_model_fn is not None:
        delete_model_fn()
    frame = pd.DataFrame(
        {
            "prediction": predictions,
            "transcript": transcripts,
            "wer": [wer(p, t) for p, t in zip(predictions, transcripts)]
        }
    )
    # sample 10 random rows and print the prediction and transcript
    print("Sample predictions:")
    sample_frame = frame.sample(10)[['prediction', 'transcript', 'wer']]
    for i, row in sample_frame.iterrows():
        print(f"Prediction: {row['prediction']}")
        print(f"Transcript: {row['transcript']}")
        print(f"WER: {row['wer']}")
        print("-----")

    
    # print the median WER:
    print(f"Median WER: {frame['wer'].median()}")
    return frame

@click.command()
@click.argument('transcription_model', 
                type=click.Choice(['whisper-large', 'whisper-large-v2', 'whisper-large-v3', 'mms', 'qwen', 'owsm', 
                                   'all']))
def transcribe_spice(transcription_model):
    # make a dummy transcription function that just returns 'dummy transcript'
    participant_id = 'VF19C'
    all_frames = []
    if transcription_model == 'all':
        for model in ['mms', 'qwen', 'owsm', 'whisper-large', 'whisper-large-v2', 'whisper-large-v3']:
            print(f"Transcribing with model {model}")
            result_frame = transcribe_valid_snippets(model, dtype=np.float64, participant_id=participant_id)
            result_frame['model'] = model
            all_frames.append(result_frame)
    final_frame = pd.concat(all_frames, ignore_index=True)
    ipdb.set_trace()

# TODO: implement this.

def _get_participant_textgrid(participant_id):
    files = os.listdir(SPICE_DIRNAME)
    # return the file that has the {participant_id} and also .TextGrid at the end
    target_file = None
    for file in files:
        if participant_id in file and file.endswith('.TextGrid'):
            assert target_file is None, f"Multiple TextGrid files found for participant {participant_id}"
            target_file = file
    assert target_file is not None, f"No TextGrid file found for participant {participant_id}"
    return os.path.join(SPICE_DIRNAME, target_file)

def get_annotated_textgrid(participant_id):
    filename = list(filter(lambda x: participant_id in x and x.endswith('is_valid_annotated.TextGrid'), os.listdir(SPICE_DIRNAME)))[0]
    return os.path.join(SPICE_DIRNAME, filename)

def get_participant_wav_file(participant_id):
    files = os.listdir(SPICE_DIRNAME)
    # return the file that has the {participant_id} and also .wav at the end
    target_file = None
    for file in files:
        if participant_id in file and file.endswith('.wav'):
            assert target_file is None, f"Multiple wav files found for participant {participant_id}"
            target_file = file
    assert target_file is not None, f"No wav file found for participant {participant_id}"
    return os.path.join(SPICE_DIRNAME, target_file)

@click.command()
@click.argument('participant_id') 
def create_is_valid_textgrid_tier(participant_id):
    # Path to your existing TextGrid
    input_path = _get_participant_textgrid(participant_id)
    output_path = input_path.replace('.TextGrid', '_with_is_valid.TextGrid')

    # Read the TextGrid
    tg = textgrid.openTextgrid(input_path, includeEmptyIntervals=True)

    # Choose the source tier (the one whose intervals you want to copy)
    source_tier_name = "utterance"
    source_tier = tg.getTier(source_tier_name)

    # Create a new tier with the same intervals but empty labels
    new_tier_name = f"is-valid"
    new_entries = [(start, end, "") for start, end, _ in source_tier.entries]

    # Create and add the new tier
    new_tier = textgrid.IntervalTier(new_tier_name, new_entries, tg.minTimestamp, tg.maxTimestamp)
    tg.addTier(new_tier)

    # Save the modified TextGrid
    tg.save(output_path, format='long_textgrid', includeBlankSpaces=True)
    print(f"New tier '{new_tier_name}' added and saved to {output_path}")

@click.command()
@click.argument('participant_id') 
def inspect_annotation_distribution(participant_id):
    # Path to your existing TextGrid
    input_path = get_annotated_textgrid(participant_id)
    annotation_entries = textgrid.openTextgrid(input_path, includeEmptyIntervals=True).getTier('is-valid').entries
    total_entries = len(annotation_entries)
    valid_entries = 0
    for start, end, label in annotation_entries:
        if 'Y' in label.strip():
            valid_entries += 1
    print(f"Valid/Total: {valid_entries}/{total_entries} {valid_entries/total_entries}")

@click.group()
def main():
    pass

main.add_command(transcribe_spice)
main.add_command(create_is_valid_textgrid_tier)
main.add_command(inspect_annotation_distribution)

if __name__ == '__main__':
    main()