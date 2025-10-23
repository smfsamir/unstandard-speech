#!/usr/bin/env python3

import torch
# from faster_whisper import WhisperModel

import sys
import os
from tempfile import NamedTemporaryFile
from dotenv import dotenv_values
from transformers import WhisperForConditionalGeneration, AutoProcessor

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

_model_size = "large-v1"  # other options: large-v1, large-v2, large-v3
_model = None

HF_CACHE_DIR = dotenv_values(".env")["HF_CACHE_DIR"]

class WhisperWrapper():

    def __init__(self, model, processor):
        self.model = model
        self.processor = processor
    
    def transcribe_from_array(self, 
                              wav_array, 
                              language="en"):
        input_features = self.processor(
            wav_array, sampling_rate=16000, return_tensors="pt"
        ).input_features.to('cuda')
        generated_ids = self.model.generate(input_features, language='en')
        transcription = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        return transcription

def get_model(model_size):
    global _model, _model_size

    if _model is not None and model_size == _model_size:
        return _model
    elif _model is not None:
        del _model
        torch.cuda.empty_cache()

    _model_size = model_size
    generation_model = WhisperForConditionalGeneration.from_pretrained(f"openai/{_model_size}", cache_dir=HF_CACHE_DIR).to('cuda')
    processor = AutoProcessor.from_pretrained(f"openai/{_model_size}", cache_dir=HF_CACHE_DIR)
    _model = WhisperWrapper(generation_model, processor)

    return _model


def whisper_transcribe(input_path, model="small", language="en"):
    return get_model(model).transcribe(input_path, language=language)


def whisper_transcribe_timestamped(input_path, model="small", language="en"):
    return get_model(model).transcribe(
        input_path, language=language, word_timestamps=True
    )


def whisper_transcribe_from_array(wav_array, model="small", language="en"):
    return get_model(f"whisper-{model}").transcribe_from_array(
        wav_array, language=language
    )

