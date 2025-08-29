#!/usr/bin/env python3
"""
Train emotion recognition model on RAVDESS Actor_01 dataset
"""
import sys
import os
import argparse

# Add src to path
sys.path.append('src')
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

def main():
    parser = argparse.ArgumentParser(description="Train emotion recognition model on RAVDESS dataset")
    parser.add_argument('--data-dir', default='data/Actor_01', help='Path to RAVDESS Actor_01 directory')
    parser.add_argument('--output-dir', default='models/emotion_ravdess_actor01', help='Output directory for trained model')
    parser.add_argument('--epochs', type=int, default=15, help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=4, help='Batch size for training')
    parser.add_argument('--learning-rate', type=float, default=1e-5, help='Learning rate')
    parser.add_argument('--test-size', type=float, default=0.2, help='Test set size (0.0-1.0)')
    parser.add_argument('--val-size', type=float, default=0.1, help='Validation set size (0.0-1.0)')
    
    args = parser.parse_args()
    
    print("🎭 RAVDESS Emotion Recognition Training")
    print("=" * 50)
    print(f"Data directory: {args.data_dir}")
    print(f"Output directory: {args.output_dir}")
    print(f"Training epochs: {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {args.learning_rate}")
    print()
    
    # Check if data directory exists
    if not os.path.exists(args.data_dir):
        print(f"❌ Data directory not found: {args.data_dir}")
        print("Please make sure you have the RAVDESS Actor_01 data in the correct location.")
        sys.exit(1)
    
    # Count audio files
    wav_files = [f for f in os.listdir(args.data_dir) if f.endswith('.wav')]
    if len(wav_files) == 0:
        print(f"❌ No WAV files found in {args.data_dir}")
        sys.exit(1)
    
    print(f"✅ Found {len(wav_files)} audio files")
    
    try:
        # Import and run training
        from emotion_trainer import EmotionTrainer
        
        # Initialize trainer
        trainer = EmotionTrainer(data_dir=args.data_dir)
        
        # Run training
        model_trainer, results = trainer.run_full_training(
            test_size=args.test_size,
            val_size=args.val_size,
            num_epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            output_dir=args.output_dir
        )
        
        print("\n🎉 Training completed successfully!")
        print(f"✅ Final accuracy: {results['accuracy']:.4f}")
        print(f"✅ Model saved to: {args.output_dir}")
        
        # Show per-emotion performance
        print("\n📊 Per-Emotion Performance:")
        report = results['classification_report']
        for emotion in trainer.dataset_info['emotions']:
            if emotion in report:
                precision = report[emotion]['precision']
                recall = report[emotion]['recall']
                f1 = report[emotion]['f1-score']
                print(f"  {emotion:>10}: Precision={precision:.3f}, Recall={recall:.3f}, F1={f1:.3f}")
        
        print(f"\n📈 Confusion matrix and plots saved to: {args.output_dir}")
        print("\n🚀 To use the trained model, update your pipeline configuration or run:")
        print(f"   python src/emotion_recognizer_trained.py")
        
        return True
        
    except Exception as e:
        print(f"❌ Training failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)