import sounddevice as sd

sd.default.samplerate = 44100
sd.default.channels = 2

myrecording = sd.rec(int(5 * 44100), blocking=True)
sd.play(myrecording, blocking=True)
