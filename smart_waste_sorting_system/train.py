#!/usr/bin/env python3
"""
Training script for Smart Waste Sorting System
"""

import os
import sys
import argparse
import json
from pathlib import Path

# Add src directory to path
sys.path.append(str(Path(__file__).parent / 'src'))

from config import Config
from data_preprocessing import DataPreprocessor
from model import WasteClassificationModel

def main():
    parser = argparse.ArgumentParser(description='Train waste classification model')
    parser.add_argument('--data_dir', type=str, required=True,
                       help='Path to raw data directory')
    parser.add_argument('--model_type', type=str, default='efficientnet',
                       choices=['efficientnet', 'resnet', 'mobilenet', 'custom'],
                       help='Type of model to train')
    parser.add_argument('--epochs', type=int, default=50,
                       help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size for training')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                       help='Learning rate')
    
    args = parser.parse_args()
    
    # Initialize configuration
    config = Config()
    config.EPOCHS = args.epochs
    config.BATCH_SIZE = args.batch_size
    config.LEARNING_RATE = args.learning_rate
    
    print("=" * 60)
    print("Smart Waste Sorting System - Model Training")
    print("=" * 60)
    
    # Step 1: Data Preprocessing
    print("\n1. Data Preprocessing...")
    preprocessor = DataPreprocessor()
    
    if not os.path.exists(args.data_dir):
        print(f"Error: Data directory {args.data_dir} does not exist")
        return
    
    try:
        metadata = preprocessor.organize_data(args.data_dir, config.PROCESSED_DATA_DIR)
        print(f"✓ Data preprocessing completed")
        print(f"  - Total images: {metadata['total_images']}")
        print(f"  - Train: {metadata['train_images']}")
        print(f"  - Validation: {metadata['validation_images']}")
        print(f"  - Test: {metadata['test_images']}")
        print(f"  - Classes: {metadata['classes']}")
    except Exception as e:
        print(f"✗ Data preprocessing failed: {str(e)}")
        return
    
    # Step 2: Model Training
    print("\n2. Model Training...")
    try:
        model = WasteClassificationModel(model_type=args.model_type)
        model.build_model()
        
        print(f"✓ Model built successfully ({args.model_type})")
        print(f"  - Input shape: {config.IMAGE_SIZE}")
        print(f"  - Number of classes: {len(metadata['classes'])}")
        
        # Train the model
        print("\nStarting training...")
        history = model.train()
        
        print("✓ Training completed")
        
    except Exception as e:
        print(f"✗ Model training failed: {str(e)}")
        return
    
    # Step 3: Model Evaluation
    print("\n3. Model Evaluation...")
    try:
        results = model.evaluate()
        
        print("✓ Model evaluation completed")
        print(f"  - Test Accuracy: {results['test_accuracy']:.4f}")
        print(f"  - Test Loss: {results['test_loss']:.4f}")
        print(f"  - Top-3 Accuracy: {results['test_top3_accuracy']:.4f}")
        
        # Plot training history
        model.plot_training_history()
        print("✓ Training history plotted")
        
    except Exception as e:
        print(f"✗ Model evaluation failed: {str(e)}")
        return
    
    # Step 4: Save training summary
    print("\n4. Saving training summary...")
    try:
        training_summary = {
            'model_type': args.model_type,
            'epochs': args.epochs,
            'batch_size': args.batch_size,
            'learning_rate': args.learning_rate,
            'data_metadata': metadata,
            'evaluation_results': results,
            'training_completed': True
        }
        
        with open(config.LOGS_DIR / 'training_summary.json', 'w') as f:
            json.dump(training_summary, f, indent=2)
        
        print("✓ Training summary saved")
        
    except Exception as e:
        print(f"✗ Failed to save training summary: {str(e)}")
    
    print("\n" + "=" * 60)
    print("Training completed successfully!")
    print("=" * 60)
    print(f"Model saved to: {config.MODELS_DIR / 'best_model.h5'}")
    print(f"Training logs: {config.LOGS_DIR}")
    print(f"Evaluation results: {config.LOGS_DIR / 'evaluation_results.json'}")

if __name__ == "__main__":
    main()



