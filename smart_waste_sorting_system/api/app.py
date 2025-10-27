from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import cv2
import numpy as np
import tensorflow as tf
from PIL import Image
import io
import base64
import json
import logging
from datetime import datetime
import os
from pathlib import Path

# Import our modules
import sys
sys.path.append(str(Path(__file__).parent.parent / 'src'))
from config import Config
from model import WasteClassificationModel

app = Flask(__name__)
CORS(app)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(Config.LOGS_DIR / 'api.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Global variables
model = None
config = Config()

class ModelManager:
    """Manages model loading and inference"""
    
    def __init__(self):
        self.model = None
        self.class_mapping = config.get_class_mapping()
        self.category_mapping = config.get_category_mapping()
        self.load_model()
    
    def load_model(self):
        """Load the trained model"""
        try:
            model_path = config.MODELS_DIR / 'best_model.h5'
            if model_path.exists():
                self.model = tf.keras.models.load_model(str(model_path))
                logger.info(f"Model loaded successfully from {model_path}")
            else:
                logger.error(f"Model file not found: {model_path}")
                raise FileNotFoundError("Model file not found")
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            raise e
    
    def preprocess_image(self, image_array):
        """Preprocess image for inference"""
        # Resize image
        image = cv2.resize(image_array, config.IMAGE_SIZE)
        
        # Normalize pixel values
        image = image.astype(np.float32) / 255.0
        
        # Add batch dimension
        image = np.expand_dims(image, axis=0)
        
        return image
    
    def predict(self, image_array):
        """Make prediction on image"""
        if self.model is None:
            raise ValueError("Model not loaded")
        
        # Preprocess image
        processed_image = self.preprocess_image(image_array)
        
        # Make prediction
        predictions = self.model.predict(processed_image, verbose=0)
        predicted_class_idx = np.argmax(predictions[0])
        confidence = np.max(predictions[0])
        
        # Get class names
        class_names = list(self.class_mapping.keys())
        predicted_class = class_names[predicted_class_idx]
        predicted_category = self.category_mapping[predicted_class]
        
        # Get all predictions
        all_predictions = {
            class_names[i]: float(predictions[0][i]) 
            for i in range(len(class_names))
        }
        
        return {
            'predicted_class': predicted_class,
            'predicted_category': predicted_category,
            'confidence': float(confidence),
            'all_predictions': all_predictions,
            'timestamp': datetime.now().isoformat()
        }

# Initialize model manager
try:
    model_manager = ModelManager()
    logger.info("Model manager initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize model manager: {str(e)}")
    model_manager = None

@app.route('/')
def index():
    """Main page"""
    return render_template('index.html')

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': model_manager is not None,
        'timestamp': datetime.now().isoformat()
    })

@app.route('/predict', methods=['POST'])
def predict():
    """Predict waste class from uploaded image"""
    try:
        if model_manager is None:
            return jsonify({'error': 'Model not available'}), 500
        
        # Check if image is provided
        if 'image' not in request.files:
            return jsonify({'error': 'No image provided'}), 400
        
        file = request.files['image']
        if file.filename == '':
            return jsonify({'error': 'No image selected'}), 400
        
        # Read image
        image_bytes = file.read()
        image_array = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
        
        if image is None:
            return jsonify({'error': 'Invalid image format'}), 400
        
        # Convert BGR to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Make prediction
        result = model_manager.predict(image_rgb)
        
        logger.info(f"Prediction made: {result['predicted_class']} ({result['confidence']:.3f})")
        
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Error in prediction: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/predict_base64', methods=['POST'])
def predict_base64():
    """Predict waste class from base64 encoded image"""
    try:
        if model_manager is None:
            return jsonify({'error': 'Model not available'}), 500
        
        data = request.get_json()
        if 'image' not in data:
            return jsonify({'error': 'No image provided'}), 400
        
        # Decode base64 image
        image_data = data['image']
        if image_data.startswith('data:image'):
            image_data = image_data.split(',')[1]
        
        image_bytes = base64.b64decode(image_data)
        image_array = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
        
        if image is None:
            return jsonify({'error': 'Invalid image format'}), 400
        
        # Convert BGR to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Make prediction
        result = model_manager.predict(image_rgb)
        
        logger.info(f"Prediction made: {result['predicted_class']} ({result['confidence']:.3f})")
        
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Error in prediction: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/batch_predict', methods=['POST'])
def batch_predict():
    """Predict waste class for multiple images"""
    try:
        if model_manager is None:
            return jsonify({'error': 'Model not available'}), 500
        
        if 'images' not in request.files:
            return jsonify({'error': 'No images provided'}), 400
        
        files = request.files.getlist('images')
        if not files:
            return jsonify({'error': 'No images selected'}), 400
        
        results = []
        for file in files:
            if file.filename == '':
                continue
            
            try:
                # Read image
                image_bytes = file.read()
                image_array = np.frombuffer(image_bytes, np.uint8)
                image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
                
                if image is None:
                    continue
                
                # Convert BGR to RGB
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                
                # Make prediction
                result = model_manager.predict(image_rgb)
                result['filename'] = file.filename
                results.append(result)
                
            except Exception as e:
                logger.error(f"Error processing file {file.filename}: {str(e)}")
                continue
        
        logger.info(f"Batch prediction completed: {len(results)} images processed")
        
        return jsonify({
            'results': results,
            'total_processed': len(results),
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Error in batch prediction: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/model_info', methods=['GET'])
def model_info():
    """Get model information"""
    try:
        if model_manager is None:
            return jsonify({'error': 'Model not available'}), 500
        
        return jsonify({
            'model_type': 'CNN',
            'image_size': config.IMAGE_SIZE,
            'classes': list(model_manager.class_mapping.keys()),
            'categories': list(set(model_manager.category_mapping.values())),
            'class_mapping': model_manager.class_mapping,
            'category_mapping': model_manager.category_mapping
        })
        
    except Exception as e:
        logger.error(f"Error getting model info: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/sorting_decision', methods=['POST'])
def sorting_decision():
    """Make sorting decision based on prediction"""
    try:
        data = request.get_json()
        if 'prediction' not in data:
            return jsonify({'error': 'No prediction provided'}), 400
        
        prediction = data['prediction']
        predicted_category = prediction.get('predicted_category')
        confidence = prediction.get('confidence', 0)
        
        # Define confidence threshold
        confidence_threshold = 0.7
        
        if confidence < confidence_threshold:
            return jsonify({
                'action': 'manual_review',
                'reason': f'Low confidence ({confidence:.3f} < {confidence_threshold})',
                'predicted_category': predicted_category,
                'confidence': confidence
            })
        
        # Define sorting actions
        sorting_actions = {
            'organic': 'sort_to_organic_bin',
            'inorganic': 'sort_to_inorganic_bin'
        }
        
        action = sorting_actions.get(predicted_category, 'unknown_category')
        
        return jsonify({
            'action': action,
            'predicted_category': predicted_category,
            'confidence': confidence,
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Error in sorting decision: {str(e)}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    # Create templates directory
    templates_dir = Path(__file__).parent / 'templates'
    templates_dir.mkdir(exist_ok=True)
    
    # Run the app
    app.run(
        host=config.API_HOST,
        port=config.API_PORT,
        debug=config.DEBUG
    )
