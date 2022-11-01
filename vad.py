import queue
import time
from collections import deque
from statistics import mean

import numpy as np
import sounddevice as sd
import soundfile as sf
import torch
import whisper

sd.default.channels = 1
sd.default.samplerate = 16000


def main():
    vad_model = torch.hub.load(repo_or_dir="snakers4/silero-vad", model="silero_vad")[0]
    whisper_model = whisper.load_model("base")

    input_audio_queue = queue.Queue()

    def callback(indata, frames, time, status):
        input_audio_queue.put(indata.copy())

    with sf.SoundFile(
        "./audio/audio.wav", mode="w+", samplerate=sd.default.samplerate, channels=1
    ) as file:
        with sd.InputStream(
            callback=callback,
            dtype="float32",
            blocksize=1600,
        ):
            with torch.no_grad():
                confidences = deque(maxlen=100)
                while True:
                    audio_chunk = input_audio_queue.get().squeeze()
                    file.write(audio_chunk)
                    confidence = vad_model(
                        torch.from_numpy(audio_chunk), sd.default.samplerate
                    ).item()
                    confidences.append(confidence)
                    if len(confidences) == 100 and mean(confidences) < 0.2:
                        break
                start = time.time()
                text = whisper_model.transcribe(audio_chunk)["text"]
                print(text)
                print(f"took {time.time() - start} seconds")


if __name__ == "__main__":
    main()
