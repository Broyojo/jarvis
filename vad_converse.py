import queue
import re
from collections import deque

import numpy as np
import sounddevice as sd
import torch
import whisper
from logmmse import logmmse
from TTS.utils.manage import ModelManager
from TTS.utils.synthesizer import Synthesizer

SAMPLE_RATE = 16000
CHANNELS = 1
BLOCK_SIZE = 1600
SECONDS_OF_SILENCE = 1
CONFIDENCE_BUFFER_LENGTH = int(SAMPLE_RATE / BLOCK_SIZE) * SECONDS_OF_SILENCE
VOICE_ACTIVITY_THRESHOLD = 0.5


class Transcriber:
    def __init__(self):
        self.model = whisper.load_model("base")

    def transcribe(self, audio_path):
        return self.model.transcribe(audio_path)


class Speech:
    def __init__(self):
        self.manager = ModelManager()
        (
            self.model_path,
            self.config_path,
            self.model_item,
        ) = self.manager.download_model("tts_models/zh-CN/baker/tacotron2-DDC-GST")
        self.synthesizer = Synthesizer(
            self.model_path,
            self.config_path,
            None,
            None,
            None,
            use_cuda=True,
        )

    def synthesize(self, text):
        return logmmse(
            np.array(self.synthesizer.tts(text), dtype=np.float32),
            self.synthesizer.output_sample_rate,
            output_file=None,
            initial_noise=1,
            window_size=160,
            noise_threshold=0.15,
        )


def main():
    sd.default.samplerate = SAMPLE_RATE
    sd.default.channels = CHANNELS

    vad = torch.hub.load(repo_or_dir="snakers4/silero-vad", model="silero_vad")[0]
    stt = Transcriber()
    tts = Speech()
    input = queue.Queue[np.ndarray]()

    def callback(indata, frames, time, status):
        input.put(indata.copy())

    with sd.InputStream(
        callback=callback,
        dtype="float32",
        blocksize=BLOCK_SIZE,  # at 16000 frames per second
    ):
        with torch.no_grad():
            while True:
                cfs = deque(maxlen=CONFIDENCE_BUFFER_LENGTH)
                chunks = deque()
                started = False

                while True:
                    chunk = input.get().squeeze()
                    cf = vad(torch.from_numpy(chunk), SAMPLE_RATE).item()
                    cfs.append(cf)

                    if True in [x > VOICE_ACTIVITY_THRESHOLD for x in cfs]:
                        if not started:
                            print("started!")
                            started = True
                        chunks.extend(chunk)
                    elif started == True:
                        print("done recording")
                        break

                text = stt.transcribe(np.asarray(chunks))["text"]

                text = re.sub(r"([;,.])?(\s*</SPEAKER>)", r".\2", text)

                audio = tts.synthesize(text)
                sd.play(audio, blocking=True)


if __name__ == "__main__":
    main()
