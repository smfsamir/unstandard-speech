import torch
import os
from tqdm import tqdm
import soundfile as sf
import math
from collections import Counter
import librosa
from praatio import textgrid 
from dotenv import dotenv_values
from functools import partial
from speechbrain.inference.classifiers import EncoderClassifier
from espnet2.bin.s2t_inference_language import Speech2Language
import click

from packages.lang_identify import identify_language_speechbrain, owsm_detect_language_from_array

SCRATCH_DIR = dotenv_values(".env")["SCRATCH_SAVE_DIR"]
DATASET_DIR = dotenv_values(".env")["DATASET_DIR"]
HF_CACHE_DIR = dotenv_values(".env")["HF_CACHE_DIR"]

SPICE_DIRNAME = f"{HF_CACHE_DIR}/spice"
SPICE_TMP_DIRNAME = f"{SCRATCH_DIR}/spice_temp"
TARGET_SAMPLING_RATE= 16000

MODEL_ID = "espnet/owsm_v3.1_ebf"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"  # no mps support yet


# NOTE: compute sampling rate function
# librosa.get_samplerate(path)

def compute_counts(identify_languge_fn, participant_id):
    participant_files = list(filter(lambda x: participant_id in x, os.listdir(SPICE_DIRNAME)))
    assert len(participant_files) == 2
    tg_index = [idx for idx, s in enumerate(participant_files) if 'TextGrid' in s][0]
    wav_index = (0 if tg_index == 1 else 1)
    tg = textgrid.openTextgrid(f"{SPICE_DIRNAME}/{participant_files[tg_index]}", False)
    entries = tg.getTier('utterance').entries
    data, _ = librosa.load(f"{SPICE_DIRNAME}/{participant_files[wav_index]}", sr=TARGET_SAMPLING_RATE)
    counter = Counter()
    for i in range(len(entries)):
        first_interval_start, first_interval_end = entries[i].start, entries[i].end
        label = entries[i].label
        slice = data[math.floor(first_interval_start * TARGET_SAMPLING_RATE): math.ceil(first_interval_end * TARGET_SAMPLING_RATE)]
        # print(samplerate)
        if len(label.split(' ')) > 2:
            sample = {
                "audio": {
                    "array": slice,
                    "sampling_rate": TARGET_SAMPLING_RATE
                }
            }
            prediction = identify_languge_fn(sample)['language_prediction']
            counter[prediction] += 1
    return counter

@click.command()
@click.argument('lang_id_model', type=click.Choice(['speechbrain', 'owsm']))
def main(lang_id_model):
    if lang_id_model == 'speechbrain':
        speechbrain_language_id = EncoderClassifier.from_hparams(source="speechbrain/lang-id-voxlingua107-ecapa", savedir=SCRATCH_DIR)
        identify_language = partial(identify_language_speechbrain, speechbrain_language_id)
    elif lang_id_model == 'owsm':
        s2l = Speech2Language.from_pretrained(
            model_tag=MODEL_ID,
            device=DEVICE,
            nbest=3  # return nbest prediction and probability
        )
        identify_language_owsm_partial = partial(owsm_detect_language_from_array, s2l)
        identify_language_owsm = lambda sample_dict: identify_language_owsm_partial(sample_dict['audio']['array'])
        identify_language = identify_language_owsm
    participants = ['VF20B', 'VF19B', 'VF21B', 'VF21D', 'VM21E', 'VM34A', 'VF19C', 'VF19A', 'VM20B'] # NOTE: don't forget that VF19A and VM20B is English-dominant, the other ones are not
    participant_to_percentages = {}
    all_counters = []
    for participant in tqdm(participants):
        counter = compute_counts(identify_language, participant)
        pct_english = counter['en: English'] / sum(counter.values())
        participant_to_percentages[participant] = pct_english
    print(participant_to_percentages)

if __name__ == '__main__':
    main()