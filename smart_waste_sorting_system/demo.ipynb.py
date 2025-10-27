#!/usr/bin/env python3
"""
Example usage script for Smart Waste Sorting System
"""

import os
import sys
import requests
import json
import numpy as np
from PIL import Image
import base64
from pathlib import Path

def create_sample_images():
    """Create sample images for testing"""
    print("🎨 Creating sample test images...")
    
    # Create sample directory
    sample_dir = Path("sample_images")
    sample_dir.mkdir(exist_ok=True)
    
    # Create sample images for each class
    classes = ["Banana_Peel", "Orange_Peel", "Plastic", "Paper", "Wood"]
    colors = {
        "Banana_Peel": (255, 255, 0),      # Yellow
        "Orange_Peel": (255, 165, 0),     # Orange
        "Plastic": (0, 255, 255),         # Cyan
        "Paper": (255, 255, 255),          # White
        "Wood": (139, 69, 19)             # Brown
    }
    
    for class_name in classes:
        # Create a simple colored image
        image = np.full((224, 224, 3), colors[class_name], dtype=np.uint8)
        
        # Add some texture/noise
        noise = np.random.randint(-50, 50, (224, 224, 3))
        image = np.clip(image + noise, 0, 255).astype(np.uint8)
        
        # Save image
        image_path = sample_dir / f"{class_name.lower()}_sample.jpg"
        Image.fromarray(image).save(image_path)
        print(f"✓ Created: {image_path}")
    
    return sample_dir

def test_prediction_with_sample(image_path, class_name):
    """Test prediction with a sample image"""
    print(f"\n🔍 Testing prediction for {class_name}...")
    
    try:
        # Test file upload prediction
        with open(image_path, 'rb') as f:
            files = {'image': f}
            response = requests.post("http://localhost:5000/predict", files=files, timeout=10)
        
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Prediction successful!")
            print(f"   Expected: {class_name}")
            print(f"   Predicted: {result.get('predicted_class', 'unknown')}")
            print(f"   Category: {result.get('predicted_category', 'unknown')}")
            print(f"   Confidence: {result.get('confidence', 0):.3f}")
            
            # Check if prediction is correct
            if result.get('predicted_class') == class_name:
                print("🎯 Correct prediction!")
            else:
                print("⚠️  Prediction differs from expected class")
            
            return result
        else:
            print(f"❌ Prediction failed: {response.status_code}")
            return None
            
    except requests.exceptions.RequestException as e:
        print(f"❌ Prediction failed: {e}")
        return None

def test_sorting_decision(prediction_result):
    """Test sorting decision based on prediction"""
    if not prediction_result:
        return
    
    print(f"\n🤖 Testing sorting decision...")
    
    try:
        data = {'prediction': prediction_result}
        response = requests.post("http://localhost:5000/sorting_decision", json=data, timeout=5)
        
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Sorting decision successful!")
            print(f"   Action: {result.get('action', 'unknown')}")
            print(f"   Category: {result.get('predicted_category', 'unknown')}")
            print(f"   Confidence: {result.get('confidence', 0):.3f}")
            
            # Interpret the action
            action = result.get('action', '')
            if 'organic' in action:
                print("🌱 → Sort to organic bin")
            elif 'inorganic' in action:
                print("♻️  → Sort to inorganic bin")
            elif 'manual' in action:
                print("👤 → Requires manual review")
            
            return result
        else:
            print(f"❌ Sorting decision failed: {response.status_code}")
            return None
            
    except requests.exceptions.RequestException as e:
        print(f"❌ Sorting decision failed: {e}")
        return None

def demonstrate_workflow():
    """Demonstrate the complete workflow"""
    print("=" * 60)
    print("🗑️  Smart Waste Sorting System - Demo")
    print("=" * 60)
    
    # Check if API is running
    try:
        response = requests.get("http://localhost:5000/health", timeout=5)
        if response.status_code != 200:
            print("❌ API server is not running!")
            print("Please start the server first: python start.py")
            return
    except requests.exceptions.RequestException:
        print("❌ API server is not running!")
        print("Please start the server first: python start.py")
        return
    
    print("✅ API server is running!")
    
    # Create sample images
    sample_dir = create_sample_images()
    
    # Test each class
    classes = ["Banana_Peel", "Orange_Peel", "Plastic", "Paper", "Wood"]
    results = []
    
    for class_name in classes:
        image_path = sample_dir / f"{class_name.lower()}_sample.jpg"
        
        # Test prediction
        prediction = test_prediction_with_sample(image_path, class_name)
        
        if prediction:
            # Test sorting decision
            sorting_decision = test_sorting_decision(prediction)
            
            results.append({
                'class': class_name,
                'prediction': prediction,
                'sorting_decision': sorting_decision
            })
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 Demo Results Summary")
    print("=" * 60)
    
    organic_count = 0
    inorganic_count = 0
    
    for result in results:
        class_name = result['class']
        prediction = result['prediction']
        sorting = result['sorting_decision']
        
        if prediction:
            predicted_class = prediction.get('predicted_class', 'unknown')
            category = prediction.get('predicted_category', 'unknown')
            confidence = prediction.get('confidence', 0)
            
            print(f"📦 {class_name}:")
            print(f"   Predicted: {predicted_class} ({category})")
            print(f"   Confidence: {confidence:.3f}")
            
            if sorting:
                action = sorting.get('action', 'unknown')
                print(f"   Action: {action}")
                
                if 'organic' in action:
                    organic_count += 1
                elif 'inorganic' in action:
                    inorganic_count += 1
            
            print()
    
    print(f"📈 Summary:")
    print(f"   Organic items: {organic_count}")
    print(f"   Inorganic items: {inorganic_count}")
    print(f"   Total processed: {len(results)}")
    
    # Clean up sample images
    print(f"\n🧹 Cleaning up sample images...")
    for file in sample_dir.glob("*.jpg"):
        file.unlink()
    sample_dir.rmdir()
    print("✅ Cleanup completed!")
    
    print("\n" + "=" * 60)
    print("🎉 Demo completed successfully!")
    print("=" * 60)

if __name__ == "__main__":
    demonstrate_workflow()
