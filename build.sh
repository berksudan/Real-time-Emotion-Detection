#!/bin/bash

# Change current directory to project directory. 
CURRENT_PATH="$( cd "$(dirname "$0")" || exit ; pwd -P )"
cd "$CURRENT_PATH" || exit

# Install python3-venv package if not installed.
sudo apt install python3-venv

# Create virtual environment directory
python3 -m venv venv/

# Activate virtual environment
source venv/bin/activate

# Upgrade Python 
python -m pip install --upgrade pip

# Check version of pip
# Version must be below 18.XX and compatible with Python 3.4+
pip --version

# Install dependencies
pip install -I opencv-python==4.4.0.46
pip install -I dlib==19.21.1
pip install -I pandas==1.1.4
pip install -I scikit-learn==0.23.2
