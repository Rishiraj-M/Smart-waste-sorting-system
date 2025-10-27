#!/usr/bin/env python3
"""
Test script for Smart Waste Sorting System
"""

import os
import sys
import requests
import json
import time
from pathlib import Path
import numpy as np
from PIL import Image
import base64

def create_test_image():
    """Create a simple test image"""
    # Create a simple test image (224x224 RGB)
    test_image = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    
    # Save as temporary file
    test_path = Path("test_image.jpg")
    Image.fromarray(test_image).save(test_path)
    
    return test_path

def test_api_health():
    """Test API health endpoint"""
    print("🔍 Testing API health...")
    
    try:
        response = requests.get("http://localhost:5000/health", timeout=5)
        if response.status_code == 200:
            health_data = response.json()
            print("✅ API is healthy!")
            print(f"   Model loaded: {health_data.get('model_loaded', False)}")
            print(f"   Status: {health_data.get('status', 'unknown')}")
            return True
        else:
            print(f"❌ API health check failed: {response.status_code}")
            return False
    except requests.exceptions.RequestException as e:
        print(f"❌ API health check failed: {e}")
        return False

def test_model_info():
    """Test model info endpoint"""
    print("\n🔍 Testing model info...")
    
    try:
        response = requests.get("http://localhost:5000/model_info", timeout=5)
        if response.status_code == 200:
            model_info = response.json()
            print("✅ Model info retrieved!")
            print(f"   Classes: {model_info.get('classes', [])}")
            print(f"   Categories: {model_info.get('categories', [])}")
            print(f"   Image size: {model_info.get('image_size', [])}")
            return True
        else:
            print(f"❌ Model info failed: {response.status_code}")
            return False
    except requests.exceptions.RequestException as e:
        print(f"❌ Model info failed: {e}")
        return False

def test_prediction():
    """Test prediction endpoint"""
    print("\n🔍 Testing prediction...")
    
    try:
        # Create test image
        test_image_path = create_test_image()
        
        # Test file upload prediction
        with open(test_image_path, 'rb') as f:
            files = {'image': f}
            response = requests.post("http://localhost:5000/predict", files=files, timeout=10)
        
        if response.status_code == 200:
            result = response.json()
            print("✅ Prediction successful!")
            print(f"   Predicted class: {result.get('predicted_class', 'unknown')}")
            print(f"   Predicted category: {result.get('predicted_category', 'unknown')}")
            print(f"   Confidence: {result.get('confidence', 0):.3f}")
            return True
        else:
            print(f"❌ Prediction failed: {response.status_code}")
            if response.text:
                try:
                    error_data = response.json()
                    print(f"   Error: {error_data.get('error', 'Unknown error')}")
                except:
                    print(f"   Response: {response.text}")
            return False
            
    except requests.exceptions.RequestException as e:
        print(f"❌ Prediction failed: {e}")
        return False
    finally:
        # Clean up test image
        if 'test_image_path' in locals() and test_image_path.exists():
            test_image_path.unlink()

def test_base64_prediction():
    """Test base64 prediction endpoint"""
    print("\n🔍 Testing base64 prediction...")
    
    try:
        # Create test image
        test_image_path = create_test_image()
        
        # Convert to base64
        with open(test_image_path, 'rb') as f:
            image_data = base64.b64encode(f.read()).decode('utf-8')
        
        # Test base64 prediction
        data = {'image': f'data:image/jpeg;base64,{image_data}'}
        response = requests.post("http://localhost:5000/predict_base64", json=data, timeout=10)
        
        if response.status_code == 200:
            result = response.json()
            print("✅ Base64 prediction successful!")
            print(f"   Predicted class: {result.get('predicted_class', 'unknown')}")
            print(f"   Predicted category: {result.get('predicted_category', 'unknown')}")
            print(f"   Confidence: {result.get('confidence', 0):.3f}")
            return True
        else:
            print(f"❌ Base64 prediction failed: {response.status_code}")
            return False
            
    except requests.exceptions.RequestException as e:
        print(f"❌ Base64 prediction failed: {e}")
        return False
    finally:
        # Clean up test image
        if 'test_image_path' in locals() and test_image_path.exists():
            test_image_path.unlink()

def test_sorting_decision():
    """Test sorting decision endpoint"""
    print("\n🔍 Testing sorting decision...")
    
    try:
        # Create mock prediction result
        mock_prediction = {
            'predicted_class': 'Banana_Peel',
            'predicted_category': 'organic',
            'confidence': 0.85
        }
        
        data = {'prediction': mock_prediction}
        response = requests.post("http://localhost:5000/sorting_decision", json=data, timeout=5)
        
        if response.status_code == 200:
            result = response.json()
            print("✅ Sorting decision successful!")
            print(f"   Action: {result.get('action', 'unknown')}")
            print(f"   Category: {result.get('predicted_category', 'unknown')}")
            print(f"   Confidence: {result.get('confidence', 0):.3f}")
            return True
        else:
            print(f"❌ Sorting decision failed: {response.status_code}")
            return False
            
    except requests.exceptions.RequestException as e:
        print(f"❌ Sorting decision failed: {e}")
        return False

def main():
    """Main test function"""
    print("=" * 60)
    print("🧪 Smart Waste Sorting System - API Tests")
    print("=" * 60)
    
    # Wait a moment for server to be ready
    print("⏳ Waiting for server to be ready...")
    time.sleep(2)
    
    # Run tests
    tests = [
        ("API Health", test_api_health),
        ("Model Info", test_model_info),
        ("File Prediction", test_prediction),
        ("Base64 Prediction", test_base64_prediction),
        ("Sorting Decision", test_sorting_decision)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ {test_name} test crashed: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 Test Results Summary")
    print("=" * 60)
    
    passed = 0
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} {test_name}")
        if result:
            passed += 1
    
    print(f"\n📈 Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! The system is working correctly.")
    else:
        print("⚠️  Some tests failed. Check the logs and configuration.")
    
    print("=" * 60)

if __name__ == "__main__":
    main()



