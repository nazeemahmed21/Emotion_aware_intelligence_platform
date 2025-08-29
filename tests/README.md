# Tests Directory

This directory contains all test scripts for the Emotion-Aware Voice Pipeline.

## Test Files

- `test_whisper_direct.py` - Direct Whisper tests without temporary files
- `test_whisper_windows.py` - Windows-specific Whisper tests with file handling
- `test_simple_transcription.py` - Basic transcription functionality tests
- `test_audio_fix.py` - Audio processing and validation tests
- `test_quick.py` - Quick pipeline test with synthetic audio
- `test_create_audio.py` - Creates test audio files for debugging

## Running Tests

### Run all tests:
```bash
python run.py test
```

### Run individual tests:
```bash
python tests/test_whisper_direct.py
python tests/test_quick.py
```

### Create test audio files:
```bash
python tests/test_create_audio.py
```

## Test Structure

All tests follow the same pattern:
- Import paths are set up to access the `src/` directory
- Each test returns `True` for success, `False` for failure
- Tests can be run individually or through the main runner
- Temporary files are cleaned up automatically

## Common Issues

If tests fail:
1. Make sure all dependencies are installed: `pip install -r requirements.txt`
2. Check that Ollama is running: `ollama serve`
3. Verify models are available: `ollama list`
4. Run the setup: `python run.py setup`