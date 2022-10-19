import queue

import sounddevice as sd
import torch

sd.default.samplerate = 16000
sd.default.channels = 1

q = queue.Queue()


def callback(indata, frames, time, status):
    q.put(indata.copy())


model, _ = torch.hub.load(repo_or_dir="snakers4/silero-vad", model="silero_vad")

with sd.InputStream(
    callback=callback,
    dtype="float32",
    blocksize=1600,
):
    while True:
        chunk = q.get().squeeze()
        confidence = model(torch.from_numpy(chunk), sd.default.samplerate).item()
        print("#" * int(confidence * 100))
