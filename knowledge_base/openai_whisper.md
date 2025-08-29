1. OpenAI Whisper (Speech-to-Text)
Overview
OpenAI's Whisper is a state-of-the-art automatic speech recognition (ASR) system trained on 680,000 hours of multilingual and multitask supervised data. It's robust to accents, background noise, and technical language.

Installation
bash
pip install openai-whisper
Additional Dependencies
bash
# On Ubuntu/Debian
sudo apt update && sudo apt install ffmpeg

# On macOS
brew install ffmpeg

# On Windows
choco install ffmpeg
# or download from https://ffmpeg.org/download.html
Basic Usage
python
import whisper

# Load model (options: tiny, base, small, medium, large)
model = whisper.load_model("base")

# Transcribe audio from file
result = model.transcribe("audio.mp3")
print(result["text"])

# For non-English audio, you can specify the language
result = model.transcribe("audio.mp3", language="spanish")

# Get word-level timestamps
result = model.transcribe("audio.mp3", word_timestamps=True)
Real-time Audio Processing
For real-time processing, you'll need to capture and buffer audio:

python
import pyaudio
import wave
import numpy as np
import whisper

# Audio parameters
CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000  # Whisper works best with 16kHz
RECORD_SECONDS = 5

# Initialize Whisper
model = whisper.load_model("base")

# Initialize PyAudio
p = pyaudio.PyAudio()

stream = p.open(format=FORMAT,
                channels=CHANNELS,
                rate=RATE,
                input=True,
                frames_per_buffer=CHUNK)

print("Recording...")
frames = []

for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
    data = stream.read(CHUNK)
    frames.append(data)

print("Finished recording")

# Stop and close the stream
stream.stop_stream()
stream.close()
p.terminate()

# Save the recorded data as a WAV file
wf = wave.open("output.wav", 'wb')
wf.setnchannels(CHANNELS)
wf.setsampwidth(p.get_sample_size(FORMAT))
wf.setframerate(RATE)
wf.writeframes(b''.join(frames))
wf.close()

# Transcribe the audio
result = model.transcribe("output.wav")
print("Transcription:", result["text"])
Advanced Options
python
# Transcribe with specific options
result = model.transcribe(
    "audio.mp3",
    language="en",          # Language code
    task="transcribe",      # Or "translate" for translation to English
    temperature=0.0,        # Control randomness (0.0 to 1.0)
    best_of=5,              # Number of candidates when sampling with non-zero temperature
    beam_size=5,            # Number of beams in beam search
    patience=1.0,           # Patience value for beam search (https://arxiv.org/abs/2204.05424)
    length_penalty=1.0,     # Alpha values for length penalty
    suppress_tokens="-1",   # Comma-separated list of token IDs to suppress
    initial_prompt="Hello"  # Optional text to provide as a prompt
)
Performance Tips
Use smaller models ("tiny", "base") for faster inference

Process audio in chunks for real-time applications

Consider using Whisper.cpp for even faster performance on CPU

Error Handling
python
import whisper
import sys

try:
    model = whisper.load_model("base")
    result = model.transcribe("audio.mp3")
except Exception as e:
    print(f"Error: {e}", file=sys.stderr)
    # Fallback to alternative STT or error message
This covers the basics of using Whisper for speech-to-text in your application. Let me know when you're ready for the next component.

