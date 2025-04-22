import os
import subprocess
import sys

# List of required packages
required_packages = [
    'numpy',
    'firebase-admin',
    'flask',
    'joblib',
    'torch',
    'torchvision',
    'pillow',
    'opencv-python',
    'pyserial'
]

def install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])

for package in required_packages:
    try:
        __import__(package.replace('-', '_'))
        print(f"✅ {package} already installed.")
    except ImportError:
        print(f"📦 Installing {package}...")
        install(package)
