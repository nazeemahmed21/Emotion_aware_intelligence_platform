emotion-aware-voice-feedback-bot/
│── README.md                  # Project overview, setup, usage instructions
│── requirements.txt           # Python dependencies
│── .gitignore                 # Ignore virtualenv, cache, etc.

│── app.py                     # Main entrypoint (Streamlit/FastAPI UI)
│── config.py                  # Configurations (model paths, thresholds, etc.)

│── data/                      # (Optional) Sample audio files
│   └── samples/
│       ├── happy.wav
│       ├── sad.wav
│       └── angry.wav

│── models/                    # Pretrained or fine-tuned models
│   ├── emotion_model/         # Speech emotion recognition (SER) model
│   └── text_model/            # (Optional) sentiment/NLP model if you add text input

│── src/                       # Core project source code
│   ├── __init__.py
│   ├── audio_utils.py         # Functions for recording, saving, preprocessing audio
│   ├── emotion_recognition.py # SER pipeline (loading model + inference)
│   ├── feedback_logic.py      # Maps detected emotions → feedback messages
│   ├── text_utils.py          # (Optional) NLP sentiment analysis helpers
│   └── ui_utils.py            # Streamlit components / helper functions

│── deployment/                # Deployment-related files
│   ├── Dockerfile             # For containerization (if needed)
│   ├── docker-compose.yml     # (Optional) if you run with DB
│   └── cloud_setup.md         # Notes if deploying on HuggingFace/Render/etc.

│── tests/                     # Unit tests
│   ├── test_audio_utils.py
│   ├── test_emotion_recognition.py
│   └── test_feedback_logic.py

│── docs/                      # Documentation for your viva/presentation
│   ├── architecture.md        # Diagram + explanation of flow
│   ├── models_used.md         # Which open models you picked + why
│   └── future_work.md         # Possible improvements
