import numpy as np
import ipdb
import librosa 
import torch
import loguru

logger = loguru.logger

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"  # no mps support yet
TARGET_SAMPLE_RATE = 16000

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

def _naive_decode_long(s2t, wav_array, config, chunk_size=30 * TARGET_SAMPLE_RATE):
    predictions = []
    for chunk in range(0, len(wav_array), chunk_size):
        audio_chunk = wav_array[chunk : chunk + chunk_size]
        predictions.append(s2t(audio_chunk, **config)[0][-2])
    if config["predict_time"]:
        return [(start, end, text) for pred in predictions for start, end, text in pred]
    return " ".join(predictions)

def owsm_transcribe_from_array(
    s2t, 
    wav_array: np.ndarray,
    text_prompt: "str | None" = None,
    naive_long=True,  # the proper long-form decoding is super resource intensive on CPU
    timestamps=False,
    translate: "tuple[str, str]" = ("eng", "eng"),
):
    assert (
        wav_array.dtype == np.float64 and np.abs(wav_array).max() <= 1.0
    ), "owsm expects float64 normalized pcm array"

    # enable long-form decoding for audio longer than 30s
    long = len(wav_array) > 30 * TARGET_SAMPLE_RATE

    config = {}
    if text_prompt is not None:
        config["text_prev"] = text_prompt
    config["lang_sym"] = f"<{translate[0]}>"
    config["task_sym"] = (
        "<asr>" if translate[0] == translate[1] else f"<st_{translate[1]}>"
    )
    config["predict_time"] = timestamps

    if long:
        if naive_long:
            return _naive_decode_long(s2t, wav_array, config)
        else:
            del config["predict_time"]
            result = s2t.decode_long(wav_array, **config)
            if timestamps:
                return result
            else:
                return " ".join(res[2] for res in result)
    else:
        return s2t(wav_array, **config)[0][-2]
# def identify_language_owsm(sample):