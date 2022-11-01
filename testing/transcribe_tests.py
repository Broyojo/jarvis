import numpy as np
import speech_recognition as sr
import torch
import whisper

model = whisper.load_model("tiny")
r = sr.Recognizer()

with sr.Microphone() as source:
    r.adjust_for_ambient_noise(source)
    audio = r.listen(source)

result = model.transcribe(torch.Tensor(audio.get_wav_data(convert_rate=16_000)))
print(result)

from whisper import Whisper, load_model
