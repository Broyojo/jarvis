import numpy as np
import torch
import whisper


class TranscribeResult:
    def __init__(self, text, language):
        self.text = text
        self.language = language


class Transcriber:
    def __init__(self, model: str):
        self.model = whisper.load_model(model)

    def transcribe(self, audio: str | np.ndarray | torch.Tensor):
        t = self.model.transcribe(audio)
        return t
