#!/usr/bin/env python3
"""
Setup script for Smart Waste Sorting System
This script helps you set up the project and prepare your data
"""

import os
import sys
import shutil
from pathlib import Path

def create_project_structure():
    """Create the complete project structure"""
    print("🏗️  Creating project structure...")
    
    # Project root
    project_root = Path(__file__).parent
    
    # Directories to create
    directories = [
        project_root / "data" / "raw",
        project_root / "data" / "processed", 
        project_root / "data" / "train",
        project_root / "data" / "validation",
        project_root / "data" / "test",
        project_root / "models",
        project_root / "logs",
        project_root / "logs" / "detections",
        project_root / "templates"
    ]
    
    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)
        print(f"✓ Created: {directory}")
    
    print("✅ Project structure created!")

def create_sample_data_structure():
    """Create sample data structure for user reference"""
    print("\n📁 Creating sample data structure...")
    
    project_root = Path(__file__).parent
    raw_data_dir = project_root / "data" / "raw"
    
    # Create sample class directories
    classes = ["Banana_Peel", "Orange_Peel", "Plastic", "Paper", "Wood"]
    
    for class_name in classes:
        class_dir = raw_data_dir / class_name
        class_dir.mkdir(exist_ok=True)
        
        # Create a README file in each class directory
        readme_content = f"""# {class_name} Images

Place your {class_name} images in this directory.

Supported formats: .jpg, .jpeg, .png

Example structure:
- image1.jpg
- image2.jpg
- image3.jpg
- ...

Total images needed: ~90 images per class (450 total)
"""
        
        readme_path = class_dir / "README.md"
        readme_path.write_text(readme_content)
        print(f"✓ Created: {class_dir}")

def create_data_organization_guide():
    """Create a guide for organizing data"""
    print("\n📋 Creating data organization guide...")
    
    guide_content = """# Data Organization Guide

## Required Data Structure

Organize your 450 labeled images in the following structure:

```
data/raw/
├── Banana_Peel/
│   ├── banana_001.jpg
│   ├── banana_002.jpg
│   └── ... (90 images)
├── Orange_Peel/
│   ├── orange_001.jpg
│   ├── orange_002.jpg
│   └── ... (90 images)
├── Plastic/
│   ├── plastic_001.jpg
│   ├── plastic_002.jpg
│   └── ... (90 images)
├── Paper/
│   ├── paper_001.jpg
│   ├── paper_002.jpg
│   └── ... (90 images)
└── Wood/
    ├── wood_001.jpg
    ├── wood_002.jpg
    └── ... (90 images)
```

## Image Requirements

- **Format**: JPG, JPEG, or PNG
- **Size**: Any size (will be resized to 224x224)
- **Quality**: Clear, well-lit images
- **Content**: Single waste item per image
- **Background**: Various backgrounds recommended

## Data Distribution

- **Total Images**: 450
- **Per Class**: ~90 images
- **Train/Validation/Test**: 60%/20%/20% split (automatic)

## Tips for Better Results

1. **Variety**: Include different angles, lighting conditions, and backgrounds
2. **Quality**: Use clear, high-resolution images
3. **Consistency**: Ensure consistent labeling
4. **Balance**: Try to have similar number of images per class

## Next Steps

1. Place your images in the appropriate class directories
2. Run: `python train.py --data_dir data/raw`
3. Start the API: `python start.py`
"""
    
    guide_path = Path(__file__).parent / "DATA_ORGANIZATION_GUIDE.md"
    guide_path.write_text(guide_content)
    print(f"✓ Created: {guide_path}")

def create_quick_start_script():
    """Create a quick start script"""
    print("\n🚀 Creating quick start script...")
    
    script_content = """#!/usr/bin/env python3
\"\"\"
Quick Start Script for Smart Waste Sorting System
\"\"\"

import subprocess
import sys
import os
from pathlib import Path

def main():
    print("🗑️  Smart Waste Sorting System - Quick Start")
    print("=" * 50)
    
    # Check if data exists
    data_dir = Path("data/raw")
    if not data_dir.exists() or not any(data_dir.iterdir()):
        print("❌ No data found in data/raw/")
        print("Please organize your images first:")
        print("1. Create class directories: Banana_Peel, Orange_Peel, Plastic, Paper, Wood")
        print("2. Place ~90 images in each directory")
        print("3. Run this script again")
        return
    
    # Check if model exists
    model_path = Path("models/best_model.h5")
    if not model_path.exists():
        print("🤖 No trained model found. Starting training...")
        try:
            subprocess.run([sys.executable, "train.py", "--data_dir", "data/raw"], check=True)
            print("✅ Training completed!")
        except subprocess.CalledProcessError:
            print("❌ Training failed. Check your data and try again.")
            return
    else:
        print("✅ Trained model found!")
    
    # Start the API server
    print("🌐 Starting API server...")
    print("📱 Web Interface: http://localhost:5000")
    print("🔗 API Health: http://localhost:5000/health")
    print("Press Ctrl+C to stop")
    
    try:
        subprocess.run([sys.executable, "api/app.py"], check=True)
    except KeyboardInterrupt:
        print("\\n🛑 Server stopped")
    except subprocess.CalledProcessError:
        print("❌ Failed to start server")

if __name__ == "__main__":
    main()
"""
    
    script_path = Path(__file__).parent / "quick_start.py"
    script_path.write_text(script_content)
    print(f"✓ Created: {script_path}")

def main():
    """Main setup function"""
    print("=" * 60)
    print("🗑️  Smart Waste Sorting System - Setup")
    print("=" * 60)
    
    # Create project structure
    create_project_structure()
    
    # Create sample data structure
    create_sample_data_structure()
    
    # Create data organization guide
    create_data_organization_guide()
    
    # Create quick start script
    create_quick_start_script()
    
    print("\n" + "=" * 60)
    print("✅ Setup completed successfully!")
    print("=" * 60)
    print("\n📋 Next Steps:")
    print("1. 📁 Organize your 450 images in data/raw/ (see DATA_ORGANIZATION_GUIDE.md)")
    print("2. 🤖 Train the model: python train.py --data_dir data/raw")
    print("3. 🚀 Start the system: python start.py")
    print("4. 🌐 Open http://localhost:5000 in your browser")
    print("\n📚 Documentation:")
    print("- README.md: Complete documentation")
    print("- DATA_ORGANIZATION_GUIDE.md: Data organization guide")
    print("- quick_start.py: Automated setup and start")
    print("\n🔧 Testing:")
    print("- python test_api.py: Test the API endpoints")
    print("\n" + "=" * 60)

if __name__ == "__main__":
    main()



