#!/usr/bin/env python3

import torch
from faster_whisper import WhisperModel

import sys
import os
from tempfile import NamedTemporaryFile

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from .audio import audio_array_to_wav_file

_model_size = "large-v1"  # other options: large-v1, large-v2, large-v3
_model = None

def get_model(model_size):
    global _model, _model_size

    if _model is not None and model_size == _model_size:
        return _model

    _model_size = model_size
    if torch.cuda.is_available():
        # Run on GPU with FP16
        _model = WhisperModel(model_size, device="cuda", compute_type="float16")
    else:
        # Run on CPU with INT8
        _model = WhisperModel(model_size, device="cpu", compute_type="int8")

    return _model


def whisper_transcribe(input_path, model="small", language="en"):
    return get_model(model).transcribe(input_path, language=language)


def whisper_transcribe_timestamped(input_path, model="small", language="en"):
    return get_model(model).transcribe(
        input_path, language=language, word_timestamps=True
    )


def whisper_transcribe_from_array(wav_array, model="small", language="en"):
    with NamedTemporaryFile(suffix=".wav") as f:
        audio_array_to_wav_file(wav_array, f.name)
        return whisper_transcribe(f.name, model=model, language=language)


def whisper_transcribe_timestamped_from_array(wav_array, model="small", language="en"):
    with NamedTemporaryFile(suffix=".wav") as f:
        audio_array_to_wav_file(wav_array, f.name)
        return whisper_transcribe_timestamped(f.name, model=model, language=language)



def whisper_output_to_text(output, timestamped=False):
    if hasattr(output, "text"):
        return getattr(output, "text")
    if type(output) == dict and "text" in output:
        return output["text"]
    segments, _ = output
    if timestamped:
        return " ".join(word.word for segment in segments for word in segment.words)
    else:
        return " ".join(segment.text for segment in segments)