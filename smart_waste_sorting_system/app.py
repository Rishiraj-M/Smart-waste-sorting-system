from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import base64
import io
import json
import os
import sys
from pathlib import Path
import logging
from datetime import datetime

# Add the src directory to the path to import config
sys.path.append(str(Path(__file__).parent.parent / 'src'))
from config import Config

app = Flask(__name__, 
           template_folder='web',
           static_folder='web',
           static_url_path='')
CORS(app)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Industry mapping data (this would typically come from Linear MCP)
INDUSTRY_MAPPING = {
    'Banana_Peel': {
        'category': 'organic',
        'industry_type': 'Composting & Agriculture',
        'applications': 'Used in composting facilities, biogas production, and organic fertilizer manufacturing. High nutrient content makes it valuable for soil enrichment.'
    },
    'Orange_Peel': {
        'category': 'organic',
        'industry_type': 'Food Processing & Cosmetics',
        'applications': 'Extracted for essential oils, pectin production, and natural flavoring. Used in cosmetics, cleaning products, and food additives.'
    },
    'Plastic': {
        'category': 'inorganic',
        'industry_type': 'Recycling & Manufacturing',
        'applications': 'Processed into new plastic products, textile fibers, construction materials, and packaging. Various plastic types have different recycling applications.'
    },
    'Paper': {
        'category': 'inorganic',
        'industry_type': 'Pulp & Paper Industry',
        'applications': 'Recycled into new paper products, cardboard, insulation materials, and packaging. Reduces deforestation and energy consumption.'
    },
    'Wood': {
        'category': 'inorganic',
        'industry_type': 'Construction & Furniture',
        'applications': 'Used for particle board, mulch, biomass fuel, and construction materials. Can be processed into engineered wood products.'
    }
}

class WasteDetectionService:
    """Service for waste detection and industry mapping"""
    
    def __init__(self):
        self.config = Config()
        self.category_mapping = self.config.get_category_mapping()
        
    def detect_waste_items(self, image_data):
        """
        Simulate waste detection from image data
        In a real implementation, this would use a trained ML model
        """
        # For demo purposes, return mock detection results
        # In production, this would process the actual image
        
        # Simulate different detection scenarios
        import random
        
        scenarios = [
            # Scenario 1: Single organic item
            [{'class': 'Banana_Peel', 'confidence': 0.92, 'bbox': [100, 150, 200, 250]}],
            # Scenario 2: Multiple items
            [
                {'class': 'Plastic', 'confidence': 0.88, 'bbox': [50, 100, 150, 200]},
                {'class': 'Paper', 'confidence': 0.85, 'bbox': [200, 120, 300, 220]}
            ],
            # Scenario 3: Organic waste
            [{'class': 'Orange_Peel', 'confidence': 0.90, 'bbox': [120, 180, 220, 280]}],
            # Scenario 4: Mixed waste
            [
                {'class': 'Wood', 'confidence': 0.87, 'bbox': [80, 140, 180, 240]},
                {'class': 'Banana_Peel', 'confidence': 0.83, 'bbox': [250, 160, 350, 260]}
            ],
            # Scenario 5: No detection
            []
        ]
        
        # Randomly select a scenario for demo
        detections = random.choice(scenarios)
        
        # Add category information
        for detection in detections:
            detection['category'] = self.category_mapping.get(detection['class'], 'unknown')
        
        return detections
    
    def get_industry_applications(self, detections):
        """Get industry applications for detected waste items"""
        applications = []
        
        for detection in detections:
            waste_type = detection['class']
            if waste_type in INDUSTRY_MAPPING:
                app_info = INDUSTRY_MAPPING[waste_type].copy()
                app_info['waste_type'] = waste_type
                applications.append(app_info)
        
        return applications

# Initialize the detection service
detection_service = WasteDetectionService()

@app.route('/')
def index():
    """Serve the main webpage"""
    return render_template('index.html')

@app.route('/api/detect-waste', methods=['POST'])
def detect_waste():
    """API endpoint for waste detection"""
    try:
        data = request.get_json()
        
        if not data or 'image' not in data:
            return jsonify({'error': 'No image data provided'}), 400
        
        # Extract image data (base64 encoded)
        image_data = data['image']
        
        # Remove data URL prefix if present
        if image_data.startswith('data:image'):
            image_data = image_data.split(',')[1]
        
        # Decode base64 image
        try:
            image_bytes = base64.b64decode(image_data)
            logger.info(f"Received image data: {len(image_bytes)} bytes")
        except Exception as e:
            logger.error(f"Error decoding image: {e}")
            return jsonify({'error': 'Invalid image data'}), 400
        
        # Perform waste detection
        detections = detection_service.detect_waste_items(image_bytes)
        
        # Get industry applications
        industry_applications = detection_service.get_industry_applications(detections)
        
        # Prepare response
        response = {
            'detections': detections,
            'industry_applications': industry_applications,
            'timestamp': datetime.now().isoformat(),
            'total_detected': len(detections)
        }
        
        logger.info(f"Detection completed: {len(detections)} items found")
        return jsonify(response)
        
    except Exception as e:
        logger.error(f"Error in detect_waste: {e}")
        return jsonify({'error': 'Internal server error'}), 500

@app.route('/api/industry-info/<waste_type>', methods=['GET'])
def get_industry_info(waste_type):
    """Get industry information for a specific waste type"""
    try:
        if waste_type in INDUSTRY_MAPPING:
            info = INDUSTRY_MAPPING[waste_type].copy()
            info['waste_type'] = waste_type
            return jsonify(info)
        else:
            return jsonify({'error': 'Waste type not found'}), 404
            
    except Exception as e:
        logger.error(f"Error getting industry info: {e}")
        return jsonify({'error': 'Internal server error'}), 500

@app.route('/api/stats', methods=['GET'])
def get_stats():
    """Get system statistics"""
    try:
        stats = {
            'total_detections': 0,  # This would be tracked in a real system
            'organic_count': 0,
            'inorganic_count': 0,
            'recycling_rate': 0,
            'system_uptime': 'Active',
            'last_update': datetime.now().isoformat()
        }
        return jsonify(stats)
        
    except Exception as e:
        logger.error(f"Error getting stats: {e}")
        return jsonify({'error': 'Internal server error'}), 500

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'version': '1.0.0'
    })

@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Endpoint not found'}), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({'error': 'Internal server error'}), 500

if __name__ == '__main__':
    # Create necessary directories
    Config.create_directories()
    
    # Start the Flask application
    logger.info("Starting Smart Waste Sorting System API")
    logger.info(f"Configuration: {Config.API_HOST}:{Config.API_PORT}")
    logger.info(f"Debug mode: {Config.DEBUG}")
    
    app.run(
        host=Config.API_HOST,
        port=Config.API_PORT,
        debug=Config.DEBUG
    )
