# RAVDESS Dataset Documentation

## Overview

The **RAVDESS (Ryerson Audio-Visual Database of Emotional Speech and Song)** dataset contains **1440 audio files** from 24 actors, performing emotional speech with various intensities. It is widely used for **speech emotion recognition (SER)** research and projects.

For this project, we are using **Actor_01** samples, located at:

data/Actor_01/

yaml
Copy code

Each actor folder contains all audio samples for that actor.

---

## Filename Convention

Each audio file has a **7-part numerical identifier** in the filename, e.g.:

03-01-06-01-02-01-12.wav

markdown
Copy code

The **7 identifiers** represent the following attributes:

| Position | Meaning | Example Value | Description |
|----------|---------|---------------|-------------|
| 1 | Modality | 03 | `03` = audio-only, `01` = full AV, `02` = video-only |
| 2 | Vocal Channel | 01 | `01` = speech, `02` = song |
| 3 | Emotion | 06 | `01` = neutral, `02` = calm, `03` = happy, `04` = sad, `05` = angry, `06` = fearful, `07` = disgust, `08` = surprised |
| 4 | Emotional Intensity | 01 | `01` = normal, `02` = strong. *Note: no strong intensity for neutral emotion* |
| 5 | Statement | 02 | `01` = "Kids are talking by the door", `02` = "Dogs are sitting by the door" |
| 6 | Repetition | 01 | `01` = 1st repetition, `02` = 2nd repetition |
| 7 | Actor ID | 12 | `01–24` identifies the actor. Odd = male, Even = female |

---

## Example

Filename:

03-01-01-01-02-01-01.wav

makefile
Copy code

Breakdown:

- `03` → Audio-only  
- `01` → Speech  
- `01` → Neutral emotion  
- `01` → Normal intensity  
- `02` → Statement "Dogs are sitting by the door"  
- `01` → 1st repetition  
- `01` → Actor 01 (male)

---

## How to Extract Emotion Label in Python

```python
import os

filename = "03-01-06-01-02-01-12.wav"

# Split by "-"
parts = filename.split("-")

emotion_map = {
    "01": "neutral",
    "02": "calm",
    "03": "happy",
    "04": "sad",
    "05": "angry",
    "06": "fearful",
    "07": "disgust",
    "08": "surprised"
}

emotion_code = parts[2]
emotion_label = emotion_map[emotion_code]

print(emotion_label)  # Output: fearful
Notes
Only audio-only (03) files are used for SER tasks.

Actor folders are separated as Actor_01, Actor_02, etc.

Each actor has 60 audio clips (2 statements × 2 repetitions × 8 emotions × 2 intensities).

For a quick demo, you can select 2 samples per emotion from one actor.

References
https://www.kaggle.com/datasets/uwrfkaggler/ravdess-emotional-speech-audio?utm_source=chatgpt.com

https://zenodo.org/record/1188976?utm_source=chatgpt.com
