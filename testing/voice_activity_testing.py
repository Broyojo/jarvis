import matplotlib.pyplot as plt
import numpy as np
import pyaudio
import torch
from matplotlib.animation import FuncAnimation

FORMAT = pyaudio.paInt16
CHANNELS = 1
SAMPLE_RATE = 16000
CHUNK = int(SAMPLE_RATE / 10)

audio = pyaudio.PyAudio()
num_samples = 1536
stream = audio.open(
    format=FORMAT,
    channels=CHANNELS,
    rate=SAMPLE_RATE,
    input=True,
    frames_per_buffer=CHUNK,
)

device = "cuda" if torch.cuda.is_available() else "cpu"

print(f"device is {device}")

model, _ = torch.hub.load(
    repo_or_dir="snakers4/silero-vad", model="silero_vad", force_reload=True
)

model.to(device)


def int2float(sound):
    abs_max = np.abs(sound).max()
    sound = sound.astype("float32")
    if abs_max > 0:
        sound *= 1 / abs_max
    sound = sound.squeeze()  # depends on the use case
    return sound


voice_confidences = []


def detect_voice(i):
    global voice_confidences
    audio_chunk = stream.read(num_samples)
    audio_int16 = np.frombuffer(audio_chunk, np.int16)
    audio_float32 = int2float(audio_int16)
    print(audio_float32.shape)
    new_confidence = model(torch.from_numpy(audio_float32).to(device), 16000).item()
    voice_confidences.append(new_confidence)
    if len(voice_confidences) > 400:
        voice_confidences = voice_confidences[1:]
    plt.cla()
    plt.plot(voice_confidences)


ani = FuncAnimation(plt.gcf(), detect_voice, interval=50)

plt.tight_layout()
plt.show()
