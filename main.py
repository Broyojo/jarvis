import queue
import time
from collections import deque

import matplotlib.pyplot as plt
import numpy as np
import openai
import sounddevice as sd
import torch
import whisper
from logmmse import logmmse
from TTS.utils.manage import ModelManager
from TTS.utils.synthesizer import Synthesizer

SAMPLE_RATE = 16000  # 16000  # samples per second (openai whisper expects this)
BLOCKSIZE = 1600  # 1600  # samples per block
SILENCE_TIME = 2  # number of seconds to wait for silence
NUM_SILENCE_BLOCKS = int(SILENCE_TIME * SAMPLE_RATE / BLOCKSIZE)
CHANNELS = 1
VOICE_ACTIVITY_THRESHOLD = 0.9


class GPT3:
    def __init__(self, initial_message="大家好，我是李江。 你今天过得怎么样？"):
        self.initial_message = initial_message
        self.name = "Li Jiang"
        with open("key.txt", "r") as f:
            openai.api_key = f.read().strip()
        self.conversation = f"The following is a conversation in mandarin with an AI assistant named {self.name} and a chinese person. The assistant is helpful, creative, clever, and very friendly. \n\n{self.name}:{self.initial_message}"

    def respond_to_text(self, text):
        self.conversation += f"\nHuman:{text}"
        self.conversation += f"\n{self.name}:"
        response = ""
        while True:
            response = (
                openai.Completion.create(
                    model="text-davinci-002",
                    prompt=self.conversation,
                    temperature=0.7,
                    max_tokens=512,
                    top_p=1,
                    frequency_penalty=0.0,
                    presence_penalty=0.6,
                    stop=["\n"],
                )
                .choices[0]
                .text
            ).strip()
            if response != "":
                break
            print(":: Empty response, re-generating...")
        self.conversation += response
        return response


class TTS:
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

    def synth(self, text):
        text += "。"
        wavs = self.synthesizer.tts(text)
        enhanced = logmmse(
            np.array(wavs, dtype=np.float32),
            self.synthesizer.output_sample_rate,
            output_file=None,
            initial_noise=1,
            window_size=160,
            noise_threshold=0.15,
        )
        return enhanced

    def play(self, audio):
        sd.play(audio, blocking=True, samplerate=self.synthesizer.output_sample_rate)


def main():
    sd.default.samplerate = SAMPLE_RATE
    sd.default.channels = 1
    sd.default.blocksize = BLOCKSIZE

    vad = torch.hub.load(repo_or_dir="snakers4/silero-vad", model="silero_vad")[0]
    print(":: Loaded VAD model")
    stt = whisper.load_model("base")
    print(":: Loaded STT model")
    gpt3 = GPT3()
    print(":: GPT3 loaded")
    tts = TTS()

    tts.play(tts.synth(gpt3.initial_message))

    input_audio = queue.Queue()
    output_audio = queue.Queue()

    def process(indata, outdata, frames, time, status):
        if status:
            print(":: Status:", status)
        input_audio.put(indata.copy())
        try:
            outdata[:] = output_audio.get(block=False)
        except queue.Empty:
            pass

    with sd.Stream(callback=process, dtype="float32"):
        with torch.no_grad():
            # conversation loop
            while True:
                # recording
                cs = deque(maxlen=NUM_SILENCE_BLOCKS)
                audios = deque()
                print(f":: Waiting for max of {NUM_SILENCE_BLOCKS} blocks of silence")
                started = False
                while True:
                    audio = input_audio.get().squeeze()
                    c = vad(torch.from_numpy(audio), SAMPLE_RATE).item()
                    cs.append(c)
                    audios.append(np.expand_dims(audio, axis=1))
                    if any([x > VOICE_ACTIVITY_THRESHOLD for x in cs]):
                        if not started:
                            print(":: Starting recording")
                            started = True
                        else:
                            print(":: Recording in progress")
                    elif started:
                        print(":: Done recording")
                        started = False
                        cs.clear()
                        break
                # generating
                text = stt.transcribe(np.asarray(audios).flatten())["text"]
                audios.clear()
                print(f":: Transcribed text: {text}")

                response = gpt3.respond_to_text(text)
                print(f":: Response: {response}")

                synth = tts.synth(response)
                tts.play(synth)

                input_audio.queue.clear()
                output_audio.queue.clear()

                # break  # only want to try one back and forth


if __name__ == "__main__":
    main()
