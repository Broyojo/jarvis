import numpy as np
from whispercpp import Whisper

w = Whisper.from_pretrained("base.en")
output = w.transcribe_from_file("out.wav")
print(output)
