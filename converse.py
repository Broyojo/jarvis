import numpy as np
import openai
import whisper
from logmmse import logmmse
from playsound import playsound
from scipy.io.wavfile import write
from TTS.utils.manage import ModelManager
from TTS.utils.synthesizer import Synthesizer

from recorder import Recorder


class Transcriber:
    def __init__(self):
        self.model = whisper.load_model("base")
        print(":: OpenAI Whisper model loaded")
        print(
            f":: Transcriber model is {'multilingual' if self.model.is_multilingual else 'English-only'} "
            f"and has {sum(np.prod(p.shape) for p in self.model.parameters()):,} parameters."
        )

    def transcribe(self, audio_path):
        transcribed = self.model.transcribe(audio_path)
        return transcribed


class SpeechSynthesizer:
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
        print(":: Speech Synthesis model loaded")

    def synthesize(self, text):
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


class AI:
    def __init__(self, initial_message):
        with open("key.txt", "r") as f:
            openai.api_key = f.read().strip()
        self.conversation = f"The following is a conversation in mandarin with a chinese AI assistant named Li Jing and a chinese person. The assistant is helpful, creative, clever, and very friendly. \n\nLi Jing:{initial_message}"
        print(":: OpenAI GPT-3 model loaded")

    def respond_to_input(self, text):
        self.conversation += f"\nHuman:{text}"
        self.conversation += "\nLi Jing:"
        response = ""
        while True:
            response = (
                openai.Completion.create(
                    model="text-davinci-002",
                    prompt=self.conversation,
                    temperature=0.9,
                    max_tokens=150,
                    top_p=1,
                    frequency_penalty=0.0,
                    presence_penalty=0.6,
                    stop=["\n"],
                )
                .choices[0]
                .text
            )
            if response != "":
                break
            print(":: Empty response, re-generating...")
        self.conversation += response
        return response


def main():
    initial_message = "大家好，我叫李江。"

    recorder = Recorder()
    ai = AI(initial_message)
    transcriber = Transcriber()
    synthesizer = SpeechSynthesizer()

    audio = synthesizer.synthesize(initial_message)

    write("output.wav", synthesizer.synthesizer.output_sample_rate, audio)

    playsound("output.wav")

    while True:
        recorder.listen()
        text = transcriber.transcribe("recorder_output.wav")["text"]
        response = ai.respond_to_input(text).strip()
        print(response)

        audio = synthesizer.synthesize(response)
        write("output.wav", synthesizer.synthesizer.output_sample_rate, audio)
        playsound("output.wav")


if __name__ == "__main__":
    main()
