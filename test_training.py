#!/usr/bin/env python3
"""
Test script to verify training setup and run a quick training test
"""
import sys
import os

# Add src to path
sys.path.append('src')
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

def check_requirements():
    """Check if all required packages are available"""
    print("🔍 Checking Requirements")
    print("=" * 30)
    
    required_packages = [
        'torch', 'transformers', 'librosa', 'soundfile', 
        'sklearn', 'matplotlib', 'seaborn', 'pandas', 'numpy'
    ]
    
    missing = []
    for package in required_packages:
        try:
            __import__(package)
            print(f"✅ {package}")
        except ImportError:
            print(f"❌ {package}")
            missing.append(package)
    
    if missing:
        print(f"\n⚠️  Missing packages: {', '.join(missing)}")
        print("Install with: pip install torch transformers librosa soundfile scikit-learn matplotlib seaborn")
        return False
    
    print("✅ All requirements satisfied!")
    return True

def check_dataset():
    """Check if RAVDESS dataset is available"""
    print("\n📁 Checking Dataset")
    print("=" * 20)
    
    data_dir = "data/Actor_01"
    if not os.path.exists(data_dir):
        print(f"❌ Dataset directory not found: {data_dir}")
        return False
    
    wav_files = [f for f in os.listdir(data_dir) if f.endswith('.wav')]
    if len(wav_files) == 0:
        print(f"❌ No WAV files found in {data_dir}")
        return False
    
    print(f"✅ Found {len(wav_files)} audio files")
    
    # Check emotion distribution
    emotion_map = {
        "01": "neutral", "02": "calm", "03": "happy", "04": "sad",
        "05": "angry", "06": "fearful", "07": "disgust", "08": "surprised"
    }
    
    emotion_counts = {}
    for filename in wav_files:
        parts = filename.split('-')
        if len(parts) >= 3:
            emotion_code = parts[2]
            emotion = emotion_map.get(emotion_code, 'unknown')
            emotion_counts[emotion] = emotion_counts.get(emotion, 0) + 1
    
    print("📊 Emotion distribution:")
    for emotion, count in sorted(emotion_counts.items()):
        print(f"  {emotion:>10}: {count} files")
    
    return True

def test_training_components():
    """Test that training components can be imported and initialized"""
    print("\n🧪 Testing Training Components")
    print("=" * 35)
    
    try:
        from emotion_trainer import EmotionTrainer, RAVDESSDataset
        print("✅ EmotionTrainer imported")
        
        # Test trainer initialization
        trainer = EmotionTrainer(data_dir="data/Actor_01")
        print("✅ EmotionTrainer initialized")
        
        # Test dataset loading
        audio_paths, labels = trainer.load_dataset()
        print(f"✅ Dataset loaded: {len(audio_paths)} files, {len(set(labels))} emotions")
        
        return True
        
    except Exception as e:
        print(f"❌ Training components test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_trained_model_loading():
    """Test loading trained model (if available)"""
    print("\n🤖 Testing Trained Model Loading")
    print("=" * 40)
    
    try:
        from emotion_recognizer_trained import TrainedEmotionRecognizer
        
        recognizer = TrainedEmotionRecognizer()
        info = recognizer.get_model_info()
        
        print(f"✅ Model loaded: {'Trained' if info['is_trained'] else 'Pretrained'}")
        print(f"   Emotions: {info['emotions']}")
        print(f"   Device: {info['device']}")
        
        return True
        
    except Exception as e:
        print(f"❌ Trained model loading failed: {e}")
        return False

def run_quick_training_test():
    """Run a very quick training test with minimal epochs"""
    print("\n⚡ Quick Training Test")
    print("=" * 25)
    
    try:
        from emotion_trainer import EmotionTrainer
        
        trainer = EmotionTrainer(data_dir="data/Actor_01")
        
        print("🚀 Running quick training (2 epochs)...")
        model_trainer, results = trainer.run_full_training(
            num_epochs=2,  # Very quick test
            batch_size=2,
            learning_rate=1e-4,
            output_dir="models/emotion_test_quick"
        )
        
        print(f"✅ Quick training completed!")
        print(f"   Accuracy: {results['accuracy']:.4f}")
        print(f"   Model saved to: models/emotion_test_quick")
        
        return True
        
    except Exception as e:
        print(f"❌ Quick training test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all tests"""
    print("🎭 RAVDESS Training Setup Test")
    print("=" * 50)
    
    tests = [
        ("Requirements Check", check_requirements),
        ("Dataset Check", check_dataset),
        ("Training Components", test_training_components),
        ("Trained Model Loading", test_trained_model_loading),
    ]
    
    passed = 0
    for test_name, test_func in tests:
        try:
            if test_func():
                passed += 1
                print(f"\n✅ {test_name} PASSED")
            else:
                print(f"\n❌ {test_name} FAILED")
        except Exception as e:
            print(f"\n❌ {test_name} ERROR: {e}")
    
    print(f"\n🎯 Test Results: {passed}/{len(tests)} tests passed")
    
    if passed >= 3:  # Requirements, dataset, and components must pass
        print("\n🎉 Setup looks good! Ready for training.")
        
        # Ask if user wants to run quick training test
        try:
            response = input("\n🤔 Run quick training test (2 epochs)? [y/N]: ").strip().lower()
            if response in ['y', 'yes']:
                if run_quick_training_test():
                    print("\n🚀 Quick test passed! You can now run full training:")
                    print("   python train_emotion_model.py")
                else:
                    print("\n⚠️  Quick test failed, but basic setup is OK.")
        except KeyboardInterrupt:
            print("\n👋 Skipped quick training test")
        
    else:
        print("\n⚠️  Some tests failed. Please fix the issues before training.")
    
    return passed >= 3

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)