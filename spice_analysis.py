import os
from tqdm import tqdm
import soundfile as sf
import math
from collections import Counter
import librosa
from praatio import textgrid 
from dotenv import dotenv_values
from speechbrain.inference.classifiers import EncoderClassifier
from packages.lang_identify import identify_language_speechbrain

SCRATCH_DIR = dotenv_values(".env")["SCRATCH_SAVE_DIR"]
DATASET_DIR = dotenv_values(".env")["DATASET_DIR"]
HF_CACHE_DIR = dotenv_values(".env")["HF_CACHE_DIR"]

SPICE_DIRNAME = f"{HF_CACHE_DIR}/spice"
SPICE_TMP_DIRNAME = f"{SCRATCH_DIR}/spice_temp"
TARGET_SAMPLING_RATE= 16000

language_id = EncoderClassifier.from_hparams(source="speechbrain/lang-id-voxlingua107-ecapa", savedir=SCRATCH_DIR).to('cuda')

# NOTE: compute sampling rate function
# librosa.get_samplerate(path)

def compute_counts(participant_id):
    participant_files = list(filter(lambda x: participant_id in x, os.listdir(SPICE_DIRNAME)))
    assert len(participant_files) == 2
    tg = textgrid.openTextgrid(f"{SPICE_DIRNAME}/{participant_files[1]}", False)
    entries = tg.getTier('utterance').entries
    data, _ = librosa.load(f"{SPICE_DIRNAME}/{participant_files[0]}", sr=TARGET_SAMPLING_RATE)
    counter = Counter()
    for i in range(len(entries)):
        first_interval_start, first_interval_end = entries[i].start, entries[i].end
        label = entries[i].label
        slice = data[math.floor(first_interval_start * TARGET_SAMPLING_RATE): math.ceil(first_interval_end * TARGET_SAMPLING_RATE)]
        # print(samplerate)
        if len(label.split(' ')) > 2:
            fname = f"spice_{participant_id}_{i}.wav"
            sf.write(f'{SPICE_TMP_DIRNAME}/{fname}', slice, samplerate=TARGET_SAMPLING_RATE)
            signal = language_id.load_audio(f"{SPICE_TMP_DIRNAME}/{fname}")
            sample = {
                "audio": {
                    "array": signal,
                    "sampling_rate": TARGET_SAMPLING_RATE
                }
            }
            prediction = identify_language_speechbrain(sample)
            counter[prediction[3][0]] += 1
    return counter

if __name__ == '__main__':
    participants = ['VF20B', 'VF19B', 'VF21B', 'VF21D', 'VM21E', 'VM34A', 'VF19C', 'VF19A', 'VM20B'] # NOTE: don't forget that VF19A and VM20B is English-dominant, the other ones are not
    participant_to_percentages = {}
    all_counters = []
    for participant in tqdm(participants):
        counter = compute_counts(participant)
        pct_english = counter['en: English'] / sum(counter.values())
        participant_to_percentages[participant] = pct_english
    print(participant_to_percentages)