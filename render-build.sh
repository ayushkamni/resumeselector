#!/bin/bash
# Render build script for Smart AI Resume Analyzer

echo "Installing Python dependencies..."
pip install -r requirements.txt

echo "Downloading NLTK data..."
python setup_nltk.py

echo "Build complete!"
