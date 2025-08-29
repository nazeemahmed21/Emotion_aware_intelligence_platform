# 🎭 Emotion Recognition Model Training

This guide explains how to train a custom emotion recognition model on your RAVDESS Actor_01 dataset to improve emotion detection accuracy.

## 📋 Overview

The current emotion recognition uses a pretrained model that may not be perfectly calibrated for your specific use case. By training on the RAVDESS dataset, we can:

- ✅ Improve accuracy for the 8 RAVDESS emotions
- ✅ Better calibrate confidence scores
- ✅ Adapt to the specific audio characteristics of your data
- ✅ Reduce false positives and improve overall performance

## 🎯 RAVDESS Emotions

The model will be trained to recognize these 8 emotions:
1. **Neutral** - Calm, baseline emotional state
2. **Calm** - Relaxed, peaceful
3. **Happy** - Joyful, positive
4. **Sad** - Melancholy, down
5. **Angry** - Frustrated, aggressive
6. **Fearful** - Scared, anxious
7. **Disgust** - Repulsed, disgusted
8. **Surprised** - Shocked, amazed

## 🚀 Quick Start

### 1. Test Your Setup
```bash
python test_training.py
```
This will check:
- Required packages are installed
- RAVDESS dataset is available
- Training components work
- Current model status

### 2. Run Training
```bash
# Basic training (recommended)
python train_emotion_model.py

# Custom training with options
python train_emotion_model.py --epochs 20 --batch-size 8 --learning-rate 1e-5
```

### 3. Test Trained Model
```bash
python src/emotion_recognizer_trained.py
```

## 📊 Training Process

### What Happens During Training:

1. **Dataset Loading**: Loads all 56 RAVDESS Actor_01 files
2. **Data Splitting**: 
   - 70% for training
   - 10% for validation  
   - 20% for testing
3. **Model Fine-tuning**: Fine-tunes Wav2Vec2 on your specific data
4. **Evaluation**: Tests on held-out data and generates metrics
5. **Saving**: Saves the trained model and performance plots

### Expected Results:
- **Training Time**: 10-30 minutes (depending on hardware)
- **Expected Accuracy**: 70-90% (much better than pretrained)
- **Files Generated**:
  - `models/emotion_ravdess_actor01/` - Trained model
  - `confusion_matrix.png` - Performance visualization
  - `emotion_distribution.png` - Dataset overview
  - `training_info.json` - Training metadata

## 🔧 Training Options

```bash
python train_emotion_model.py --help
```

**Key Parameters:**
- `--epochs`: Number of training epochs (default: 15)
- `--batch-size`: Batch size for training (default: 4)
- `--learning-rate`: Learning rate (default: 1e-5)
- `--data-dir`: Path to RAVDESS data (default: data/Actor_01)
- `--output-dir`: Where to save the model (default: models/emotion_ravdess_actor01)

**Recommended Settings:**
- **Fast Training**: `--epochs 10 --batch-size 8`
- **High Accuracy**: `--epochs 25 --batch-size 4 --learning-rate 5e-6`
- **Quick Test**: `--epochs 5 --batch-size 2`

## 📈 Understanding Results

### Accuracy Metrics:
- **Overall Accuracy**: Percentage of correct predictions
- **Per-Emotion Precision**: How often the model is right when it predicts an emotion
- **Per-Emotion Recall**: How often the model finds the emotion when it's present
- **F1-Score**: Balanced measure of precision and recall

### Confusion Matrix:
Shows which emotions are confused with each other:
- **Diagonal**: Correct predictions (higher is better)
- **Off-diagonal**: Confusion between emotions (lower is better)

### Common Patterns:
- **Neutral vs Calm**: Often confused (both low-energy)
- **Happy vs Surprised**: May overlap (both high-energy)
- **Angry vs Disgust**: Sometimes mixed (both negative)

## 🔄 Using the Trained Model

Once training is complete, the pipeline will automatically use your trained model:

1. **Automatic Detection**: The pipeline checks for trained models first
2. **Fallback**: If no trained model, uses the pretrained one
3. **Model Info**: Check which model is loaded in the app sidebar

### Manual Testing:
```python
from src.emotion_recognizer_trained import TrainedEmotionRecognizer

# Load your trained model
recognizer = TrainedEmotionRecognizer()

# Check model info
info = recognizer.get_model_info()
print(f"Using: {'Trained' if info['is_trained'] else 'Pretrained'} model")

# Test with audio
result = recognizer.predict_emotion(audio_data)
print(f"Emotion: {result['emotion']} ({result['confidence']:.3f})")
```

## 🛠️ Troubleshooting

### Common Issues:

**1. "No WAV files found"**
- Check that `data/Actor_01/` contains the RAVDESS files
- Verify files are named like `03-01-01-01-01-01-01.wav`

**2. "CUDA out of memory"**
- Reduce batch size: `--batch-size 2`
- Use CPU: Set `CUDA_VISIBLE_DEVICES=""`

**3. "Training very slow"**
- Reduce epochs: `--epochs 10`
- Increase batch size if you have more RAM: `--batch-size 8`

**4. "Low accuracy"**
- Increase epochs: `--epochs 25`
- Lower learning rate: `--learning-rate 5e-6`
- Check data quality and distribution

**5. "Model not loading in pipeline"**
- Check that `models/emotion_ravdess_actor01/` exists
- Verify `training_info.json` is present
- Restart the Streamlit app

### Performance Tips:

**For Better Accuracy:**
- Use more training epochs (20-30)
- Lower learning rate (1e-6 to 5e-6)
- Ensure balanced data across emotions

**For Faster Training:**
- Use GPU if available
- Increase batch size (if memory allows)
- Reduce max audio length in dataset

## 📁 File Structure After Training

```
models/emotion_ravdess_actor01/
├── config.json                 # Model configuration
├── pytorch_model.bin           # Trained model weights
├── preprocessor_config.json    # Audio preprocessing config
├── training_info.json          # Training metadata and results
├── confusion_matrix.png        # Performance visualization
└── emotion_distribution.png    # Dataset overview
```

## 🎯 Next Steps

After successful training:

1. **Test in App**: Run `python run.py` and test with audio files
2. **Compare Performance**: Try the same audio with/without trained model
3. **Fine-tune**: Adjust training parameters if needed
4. **Expand Dataset**: Consider adding more actors for better generalization

## 📚 Technical Details

**Base Model**: `ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition`
**Architecture**: Wav2Vec2 + Classification Head
**Training Method**: Fine-tuning with frozen feature extractor
**Audio Processing**: 16kHz, normalized, padded/truncated to 3 seconds
**Optimization**: AdamW with linear learning rate schedule

---

🎉 **Happy Training!** Your emotion recognition should be much more accurate after training on the RAVDESS dataset.