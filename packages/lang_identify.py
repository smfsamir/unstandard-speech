import numpy as np
import librosa 
import torch

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"  # no mps support yet

def identify_language_speechbrain(language_id_model, sample):
    audio = sample["audio"]
    wav_array = audio["array"]
    sample_rate = audio["sampling_rate"]

    # speechbrain expects float32 normalized pcm array
    if wav_array.dtype != np.float32 or np.abs(wav_array).max() > 1:
        wav_array = wav_array.astype(np.float32) / 32768

    # resample to 16 kHz
    TARGET_SAMPLING_RATE = 16_000
    if sample_rate != TARGET_SAMPLING_RATE:
        wav_array = librosa.resample(
            wav_array, orig_sr=sample_rate, target_sr=TARGET_SAMPLING_RATE
        )
    # add batch dimension
    signal = torch.from_numpy(wav_array).unsqueeze(0)
    prediction = language_id_model.classify_batch(signal)  # type: ignore
    return {'language_prediction': prediction[3][0]}

def owsm_detect_language_from_array(
    owsm_model, 
    wav_array: np.ndarray,
):
    assert (
        wav_array.dtype ==np .float64 and np.abs(wav_array).max() <= 1.0
    ), "owsm expects float64 normalized pcm array"

    result = owsm_model(wav_array)
    return {'language_prediction': result[0][0]}

# def identify_language_owsm(sample):