import queue
import threading
import time

import speech_recognition as sr
import whisper

sample_rate = 44100
chunk_time = 1  # in seconds
q = queue.Queue(maxsize=-1)
model = whisper.load_model("tiny")

with sr.Microphone(sample_rate=sample_rate) as source:

    def process():
        current_conversation = ""

        while True:
            # get the next chunk off the queue
            # wait until there is at least 1 chunk on the queue
            chunk = q.get(block=True, timeout=None)

            print(f":: (process) got chunk of {len(chunk)} samples")

            result = model.transcribe(chunk)

            if result["no_speech_prob"] > 0.5:
                print(":: speech has stopped")
                break

            current_conversation += result["text"]

    # run transcription thread in background
    threading.Thread(target=process, daemon=True).start()

    while True:
        chunk = source.stream.read(int(sample_rate * chunk_time))
        print(f":: (main) read chunk with {len(chunk)} samples")

        # put the new chunk in the queue
        # don't want to block the main thread since reading and transcription happen concurrently
        q.put(chunk, block=False)
