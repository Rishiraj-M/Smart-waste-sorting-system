#!/bin/bash

echo "Smart Waste Sorting System - Unix Startup"
echo "=========================================="
echo

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "Error: Python 3 is not installed"
    echo "Please install Python 3 from your package manager"
    exit 1
fi

# Check if we're in the right directory
if [ ! -f "app.py" ]; then
    echo "Error: app.py not found"
    echo "Please run this script from the smart_waste_sorting_system directory"
    exit 1
fi

# Install dependencies if requirements.txt exists
if [ -f "requirements.txt" ]; then
    echo "Installing Python dependencies..."
    pip3 install -r requirements.txt
    if [ $? -ne 0 ]; then
        echo "Warning: Some dependencies may not have installed correctly"
        echo "Continuing anyway..."
    fi
    echo
fi

# Make start.py executable
chmod +x start.py

# Start the application
echo "Starting Smart Waste Sorting System..."
echo
python3 start.py
