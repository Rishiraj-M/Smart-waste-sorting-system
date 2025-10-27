#!/usr/bin/env python3
"""
Startup script for Smart Waste Sorting System
This script sets up and starts the Flask application with proper configuration
"""

import os
import sys
import subprocess
import webbrowser
import time
from pathlib import Path

def check_dependencies():
    """Check if required dependencies are installed"""
    try:
        import flask
        import flask_cors
        print("✓ Flask dependencies found")
        return True
    except ImportError as e:
        print(f"✗ Missing dependency: {e}")
        print("Please install dependencies with: pip install -r requirements.txt")
        return False

def check_camera_access():
    """Check if camera is available"""
    try:
        import cv2
        cap = cv2.VideoCapture(0)
        if cap.isOpened():
            cap.release()
            print("✓ Camera access available")
            return True
        else:
            print("⚠ Camera not accessible")
            return False
    except ImportError:
        print("⚠ OpenCV not installed - camera check skipped")
        return True

def create_directories():
    """Create necessary directories"""
    directories = [
        'data', 'data/raw', 'data/processed', 'data/train', 'data/test', 'data/validation',
        'models', 'logs'
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
    
    print("✓ Directories created")

def start_server():
    """Start the Flask server"""
    print("\n🚀 Starting Smart Waste Sorting System...")
    print("=" * 50)
    
    # Set environment variables
    os.environ['FLASK_ENV'] = 'development'
    os.environ['FLASK_DEBUG'] = '1'
    
    try:
        # Import and run the app
        from app import app
        
        print(f"📱 Web interface: http://localhost:5000")
        print(f"🔧 API endpoint: http://localhost:5000/api/")
        print(f"❤️  Health check: http://localhost:5000/api/health")
        print("\n💡 Tips:")
        print("   - Allow camera access when prompted in browser")
        print("   - Use Chrome/Chromium for best camera compatibility")
        print("   - Press Ctrl+C to stop the server")
        print("\n" + "=" * 50)
        
        # Open browser after a short delay
        def open_browser():
            time.sleep(2)
            webbrowser.open('http://localhost:5000')
        
        import threading
        browser_thread = threading.Thread(target=open_browser)
        browser_thread.daemon = True
        browser_thread.start()
        
        # Start the Flask app
        app.run(
            host='0.0.0.0',
            port=5000,
            debug=True,
            use_reloader=False  # Prevent double startup
        )
        
    except KeyboardInterrupt:
        print("\n\n👋 Shutting down Smart Waste Sorting System...")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Error starting server: {e}")
        sys.exit(1)

def main():
    """Main startup function"""
    print("🗑️  Smart Waste Sorting System")
    print("=" * 50)
    
    # Check if we're in the right directory
    if not Path('app.py').exists():
        print("❌ Error: app.py not found")
        print("Please run this script from the smart_waste_sorting_system directory")
        sys.exit(1)
    
    # Run checks
    if not check_dependencies():
        sys.exit(1)
    
    check_camera_access()
    create_directories()
    
    # Start the server
    start_server()

if __name__ == '__main__':
    main()