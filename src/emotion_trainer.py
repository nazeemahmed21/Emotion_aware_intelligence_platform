#!/usr/bin/env python3
"""
Emotion Recognition Model Trainer for RAVDESS Dataset
Fine-tune the emotion recognition model on Actor_01 samples
"""
import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import (
    Wav2Vec2ForSequenceClassification, 
    Wav2Vec2FeatureExtractor,
    TrainingArguments,
    Trainer
)
import librosa
import soundfile as sf
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import json
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RAVDESSDataset(Dataset):
    """Dataset class for RAVDESS emotion recognition"""
    
    def __init__(self, audio_paths, labels, feature_extractor, max_length=16000*3):
        self.audio_paths = audio_paths
        self.labels = labels
        self.feature_extractor = feature_extractor
        self.max_length = max_length
        
        # Create label to id mapping
        unique_labels = sorted(list(set(labels)))
        self.label2id = {label: i for i, label in enumerate(unique_labels)}
        self.id2label = {i: label for label, i in self.label2id.items()}
        
    def __len__(self):
        return len(self.audio_paths)
    
    def __getitem__(self, idx):
        audio_path = self.audio_paths[idx]
        label = self.labels[idx]
        
        # Load audio
        try:
            audio, sr = librosa.load(audio_path, sr=16000)
            
            # Pad or truncate to max_length
            if len(audio) > self.max_length:
                audio = audio[:self.max_length]
            else:
                audio = np.pad(audio, (0, self.max_length - len(audio)))
            
            # Extract features
            inputs = self.feature_extractor(
                audio, 
                sampling_rate=16000, 
                return_tensors="pt", 
                padding=True
            )
            
            return {
                'input_values': inputs.input_values.squeeze(),
                'labels': torch.tensor(self.label2id[label], dtype=torch.long)
            }
            
        except Exception as e:
            logger.error(f"Error loading {audio_path}: {e}")
            # Return dummy data
            return {
                'input_values': torch.zeros(self.max_length),
                'labels': torch.tensor(0, dtype=torch.long)
            }

class EmotionTrainer:
    """Trainer class for emotion recognition model"""
    
    def __init__(self, data_dir="data/Actor_01", model_name="ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition"):
        self.data_dir = data_dir
        self.model_name = model_name
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # RAVDESS emotion mapping
        self.emotion_map = {
            "01": "neutral",
            "02": "calm", 
            "03": "happy",
            "04": "sad",
            "05": "angry",
            "06": "fearful",
            "07": "disgust",
            "08": "surprised"
        }
        
        # Initialize components
        self.feature_extractor = None
        self.model = None
        self.dataset_info = None
        
    def parse_filename(self, filename):
        """Parse RAVDESS filename to extract emotion label"""
        try:
            parts = filename.split("-")
            if len(parts) >= 3:
                emotion_code = parts[2]
                return self.emotion_map.get(emotion_code, "unknown")
            return "unknown"
        except:
            return "unknown"
    
    def load_dataset(self):
        """Load and prepare the RAVDESS dataset"""
        logger.info(f"Loading dataset from {self.data_dir}")
        
        audio_paths = []
        labels = []
        
        # Get all wav files
        for filename in os.listdir(self.data_dir):
            if filename.endswith('.wav'):
                audio_path = os.path.join(self.data_dir, filename)
                emotion = self.parse_filename(filename)
                
                if emotion != "unknown":
                    audio_paths.append(audio_path)
                    labels.append(emotion)
        
        logger.info(f"Found {len(audio_paths)} audio files")
        
        # Create dataset info
        label_counts = pd.Series(labels).value_counts()
        self.dataset_info = {
            'total_files': len(audio_paths),
            'emotions': list(label_counts.index),
            'label_counts': label_counts.to_dict(),
            'files_per_emotion': dict(zip(label_counts.index, label_counts.values))
        }
        
        logger.info("Dataset distribution:")
        for emotion, count in label_counts.items():
            logger.info(f"  {emotion}: {count} files")
        
        return audio_paths, labels
    
    def prepare_model(self, num_labels):
        """Prepare the model and feature extractor"""
        logger.info(f"Loading model: {self.model_name}")
        
        # Load feature extractor
        self.feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(self.model_name)
        
        # Load model
        self.model = Wav2Vec2ForSequenceClassification.from_pretrained(
            self.model_name,
            num_labels=num_labels,
            ignore_mismatched_sizes=True
        )
        
        # Move to device
        self.model.to(self.device)
        
        logger.info(f"Model loaded with {num_labels} emotion classes")
        
    def train_model(self, train_dataset, val_dataset, output_dir="models/emotion_ravdess", 
                   num_epochs=10, batch_size=8, learning_rate=1e-5):
        """Train the emotion recognition model"""
        
        # Training arguments
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=num_epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            warmup_steps=100,
            weight_decay=0.01,
            logging_dir=f"{output_dir}/logs",
            logging_steps=10,
            eval_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            metric_for_best_model="eval_accuracy",
            greater_is_better=True,
            save_total_limit=3,
            learning_rate=learning_rate,
            lr_scheduler_type="linear",
            dataloader_num_workers=0,  # Windows compatibility
        )
        
        # Metrics function
        def compute_metrics(eval_pred):
            predictions, labels = eval_pred
            predictions = np.argmax(predictions, axis=1)
            accuracy = accuracy_score(labels, predictions)
            return {"accuracy": accuracy}
        
        # Initialize trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            compute_metrics=compute_metrics,
        )
        
        logger.info("Starting training...")
        
        # Train the model
        train_result = trainer.train()
        
        # Save the model
        trainer.save_model()
        
        logger.info("Training completed!")
        logger.info(f"Final training loss: {train_result.training_loss:.4f}")
        
        return trainer, train_result
    
    def evaluate_model(self, trainer, test_dataset):
        """Evaluate the trained model"""
        logger.info("Evaluating model...")
        
        # Get predictions
        predictions = trainer.predict(test_dataset)
        y_pred = np.argmax(predictions.predictions, axis=1)
        y_true = predictions.label_ids
        
        # Convert back to emotion labels
        id2label = test_dataset.id2label
        y_pred_labels = [id2label[pred] for pred in y_pred]
        y_true_labels = [id2label[true] for true in y_true]
        
        # Calculate metrics
        accuracy = accuracy_score(y_true, y_pred)
        report = classification_report(y_true_labels, y_pred_labels, output_dict=True)
        
        logger.info(f"Test Accuracy: {accuracy:.4f}")
        
        # Print detailed report
        print("\nDetailed Classification Report:")
        print(classification_report(y_true_labels, y_pred_labels))
        
        # Confusion matrix
        cm = confusion_matrix(y_true_labels, y_pred_labels, labels=list(id2label.values()))
        
        return {
            'accuracy': accuracy,
            'classification_report': report,
            'confusion_matrix': cm,
            'predictions': y_pred_labels,
            'true_labels': y_true_labels
        }
    
    def plot_results(self, results, save_path="models/emotion_ravdess"):
        """Plot training results and confusion matrix"""
        os.makedirs(save_path, exist_ok=True)
        
        # Plot confusion matrix
        plt.figure(figsize=(10, 8))
        emotions = list(self.dataset_info['emotions'])
        sns.heatmap(
            results['confusion_matrix'], 
            annot=True, 
            fmt='d', 
            xticklabels=emotions,
            yticklabels=emotions,
            cmap='Blues'
        )
        plt.title('Emotion Recognition Confusion Matrix')
        plt.xlabel('Predicted Emotion')
        plt.ylabel('True Emotion')
        plt.tight_layout()
        plt.savefig(f"{save_path}/confusion_matrix.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # Plot emotion distribution
        plt.figure(figsize=(12, 6))
        emotions = list(self.dataset_info['files_per_emotion'].keys())
        counts = list(self.dataset_info['files_per_emotion'].values())
        
        plt.bar(emotions, counts, color='skyblue', alpha=0.7)
        plt.title('Dataset Emotion Distribution')
        plt.xlabel('Emotion')
        plt.ylabel('Number of Files')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(f"{save_path}/emotion_distribution.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Plots saved to {save_path}")
    
    def save_training_info(self, results, save_path="models/emotion_ravdess"):
        """Save training information and results"""
        os.makedirs(save_path, exist_ok=True)
        
        training_info = {
            'timestamp': datetime.now().isoformat(),
            'dataset_info': self.dataset_info,
            'model_name': self.model_name,
            'accuracy': float(results['accuracy']),
            'classification_report': results['classification_report'],
            'label_mapping': {
                'emotion_map': self.emotion_map,
                'label2id': getattr(self, 'label2id', {}),
                'id2label': getattr(self, 'id2label', {})
            }
        }
        
        with open(f"{save_path}/training_info.json", 'w') as f:
            json.dump(training_info, f, indent=2)
        
        logger.info(f"Training info saved to {save_path}/training_info.json")
    
    def run_full_training(self, test_size=0.2, val_size=0.1, **training_kwargs):
        """Run the complete training pipeline"""
        logger.info("🚀 Starting RAVDESS Emotion Recognition Training")
        logger.info("=" * 60)
        
        # Load dataset
        audio_paths, labels = self.load_dataset()
        
        if len(audio_paths) == 0:
            raise ValueError("No audio files found in dataset directory")
        
        # Adjust split sizes for small datasets
        unique_labels = sorted(list(set(labels)))
        min_samples_per_class = min([labels.count(label) for label in unique_labels])
        
        # Ensure we have at least 1 sample per class in each split
        min_test_size = len(unique_labels)
        min_val_size = len(unique_labels)
        
        # Adjust test_size if too small
        if int(len(audio_paths) * test_size) < min_test_size:
            test_size = min_test_size / len(audio_paths)
            logger.warning(f"Adjusted test_size to {test_size:.3f} to ensure at least 1 sample per class")
        
        # Split dataset
        train_paths, test_paths, train_labels, test_labels = train_test_split(
            audio_paths, labels, test_size=test_size, random_state=42, stratify=labels
        )
        
        # Adjust val_size if too small
        if int(len(train_paths) * val_size) < min_val_size:
            val_size = min_val_size / len(train_paths)
            logger.warning(f"Adjusted val_size to {val_size:.3f} to ensure at least 1 sample per class")
        
        train_paths, val_paths, train_labels, val_labels = train_test_split(
            train_paths, train_labels, test_size=val_size, random_state=42, stratify=train_labels
        )
        
        logger.info(f"Dataset split: Train={len(train_paths)}, Val={len(val_paths)}, Test={len(test_paths)}")
        
        # Prepare model
        unique_labels = sorted(list(set(labels)))
        self.prepare_model(len(unique_labels))
        
        # Create datasets
        train_dataset = RAVDESSDataset(train_paths, train_labels, self.feature_extractor)
        val_dataset = RAVDESSDataset(val_paths, val_labels, self.feature_extractor)
        test_dataset = RAVDESSDataset(test_paths, test_labels, self.feature_extractor)
        
        # Store label mappings
        self.label2id = train_dataset.label2id
        self.id2label = train_dataset.id2label
        
        logger.info(f"Label mapping: {self.label2id}")
        
        # Train model
        trainer, train_result = self.train_model(train_dataset, val_dataset, **training_kwargs)
        
        # Evaluate model
        results = self.evaluate_model(trainer, test_dataset)
        
        # Save results
        save_path = training_kwargs.get('output_dir', 'models/emotion_ravdess')
        self.plot_results(results, save_path)
        self.save_training_info(results, save_path)
        
        logger.info("🎉 Training completed successfully!")
        logger.info(f"Final accuracy: {results['accuracy']:.4f}")
        logger.info(f"Model saved to: {save_path}")
        
        return trainer, results

def main():
    """Main training function"""
    # Initialize trainer
    trainer = EmotionTrainer(data_dir="data/Actor_01")
    
    # Run training
    try:
        model_trainer, results = trainer.run_full_training(
            num_epochs=15,
            batch_size=4,  # Smaller batch size for stability
            learning_rate=1e-5,
            output_dir="models/emotion_ravdess_actor01"
        )
        
        print("\n🎉 Training Summary:")
        print(f"✅ Final Accuracy: {results['accuracy']:.4f}")
        print(f"✅ Model saved to: models/emotion_ravdess_actor01")
        print(f"✅ Confusion matrix and plots saved")
        
        # Show per-emotion performance
        print("\n📊 Per-Emotion Performance:")
        report = results['classification_report']
        for emotion in trainer.dataset_info['emotions']:
            if emotion in report:
                precision = report[emotion]['precision']
                recall = report[emotion]['recall']
                f1 = report[emotion]['f1-score']
                print(f"  {emotion:>10}: P={precision:.3f}, R={recall:.3f}, F1={f1:.3f}")
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()