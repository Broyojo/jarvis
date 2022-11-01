import audioop
import wave
from collections import deque

import pyaudio


class Recorder:
    def __init__(self):
        self.sample_format = pyaudio.paInt16
        self.channels = 2
        self.chunk_size = 1024
        self.sampling_rate = 44100
        self.filename = "recorder_output.wav"
        self.silence_limit = 3  # in seconds
        self.threshold_intensity = 120

    def listen(self):
        # code based on https://github.com/suda/open-home/blob/master/Python/listen/listen.py
    
        p = pyaudio.PyAudio()

        stream = p.open(
            format=self.sample_format,
            channels=self.channels,
            rate=self.sampling_rate,
            input=True,
            frames_per_buffer=self.chunk_size,
            # input_device_index=10,
        )

        stream.start_stream()

        frames = []

        chunks_per_second = int(self.sampling_rate / self.chunk_size)
        silence_buffer = deque(maxlen=self.silence_limit * chunks_per_second)
        samples_buffer = deque(maxlen=self.silence_limit * self.sampling_rate)

        started = False

        while True:
            data = stream.read(self.chunk_size, exception_on_overflow=False)
            silence_buffer.append(abs(audioop.avg(data, 2)))
            samples_buffer.extend(data)

            if True in [x > self.threshold_intensity for x in silence_buffer]:
                if not started:
                    print(":: Recording...")
                    started = True
                    frames.extend(samples_buffer)
                    samples_buffer.clear()
                else:
                    frames.extend(data)
            elif started == True:
                # The limit was reached, finish capture and deliver

                print(":: Done recording")
                stream.stop_stream()
                stream.close()
                p.terminate()

                wf = wave.open(self.filename, "wb")
                wf.setnchannels(self.channels)
                wf.setsampwidth(p.get_sample_size(self.sample_format))
                wf.setframerate(self.sampling_rate)
                frames = [s.to_bytes(1, "big") for s in frames]
                # print(samples[0], samples[1])
                # quit()
                wf.writeframes(b"".join(frames))
                wf.close()

                print("Wrote audio file")

                break
