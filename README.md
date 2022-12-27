# Jarvis

## What is it?
A nice voice assistant using GPT-3

## How does it work?
Audio from the microphone is fed into a voice activity detection model. Once the person stops speaking for 2 seconds, the audio is sent to OpenAI's Whisper model for transcription. This transcribed text is appended to a conversation which is sent to GPT-3. The response created by GPT-3 is then run through a TTS model from the Coqai TTS library. The audio from the TTS model is then played through the speakers.  After this, the process restarts.

## Limitations
- Currently the model only speaks in Chinese.
- The model does not have access to actions such as using a calculator or browsing the internet.
- The model is quite slow in its responses as it has to do a lot of audio processing, running models, and making API requests to GPT-3. 
