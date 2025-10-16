import numpy as np
import scipy.io.wavfile as wavfile

def audio_dual_channel_to_mono(input_array):
    if input_array.ndim == 2 and input_array.shape[1] == 2:
        return np.mean(input_array, axis=1).astype(np.int16)
    return input_array


TARGET_SAMPLE_RATE = 16000

def audio_file_to_array(
    input_path, desired_sample_rate=TARGET_SAMPLE_RATE, output_orig_sample_rate=False
):
    rate, data = wavfile.read(input_path)
    data = audio_dual_channel_to_mono(data)
    data = audio_resample(data, rate, desired_sample_rate)
    if output_orig_sample_rate:
        return data, rate
    return data

def audio_resample(array, src_sample_rate, target_sample_rate=TARGET_SAMPLE_RATE):
    if src_sample_rate == target_sample_rate:
        return array
    return np.interp(
        np.linspace(
            0,
            len(array),
            int(len(array) * target_sample_rate / src_sample_rate),
        ),
        np.arange(len(array)),
        array,
    ).astype(np.int16)