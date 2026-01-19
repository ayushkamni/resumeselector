"""
Run Script for Smart AI Resume Analyzer
This script handles setup and running of the application
"""

import subprocess
import sys
import os

def check_python_version():
    """Check if Python version is compatible"""
    if sys.version_info < (3, 8):
        print("❌ Python 3.8 or higher is required")
        print(f"Current version: {sys.version}")
        return False
    print(f"✓ Python version: {sys.version}")
    return True

def install_dependencies():
    """Install required dependencies"""
    try:
        print("Installing dependencies...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✓ Dependencies installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install dependencies: {e}")
        return False

def download_nltk_data():
    """Download required NLTK data"""
    try:
        import nltk
        print("Downloading NLTK data...")

        # Download required NLTK packages
        packages = ['punkt', 'punkt_tab', 'stopwords', 'wordnet']
        for package in packages:
            try:
                if package == 'punkt':
                    nltk.data.find('tokenizers/punkt')
                elif package == 'punkt_tab':
                    nltk.data.find('tokenizers/punkt_tab')
                else:
                    nltk.data.find(f'corpora/{package}')
                print(f"✓ {package} already available")
            except LookupError:
                try:
                    nltk.download(package, quiet=True)
                    print(f"✓ Downloaded {package}")
                except Exception as download_error:
                    print(f"⚠️  Failed to download {package}: {download_error}")
                    continue

        print("✓ NLTK data ready")
        return True
    except Exception as e:
        print(f"❌ Failed to setup NLTK data: {e}")
        return False

def run_application():
    """Run the Streamlit application"""
    try:
        print("Starting Smart AI Resume Analyzer...")
        print("🌐 Application will open at: http://localhost:8501")
        print("Press Ctrl+C to stop the application")
        print("-" * 50)

        # Run streamlit
        subprocess.run([sys.executable, "-m", "streamlit", "run", "app.py"])

    except KeyboardInterrupt:
        print("\n👋 Application stopped by user")
    except Exception as e:
        print(f"❌ Failed to start application: {e}")

def main():
    """Main setup and run function"""
    print("🎯 Smart AI Resume Analyzer")
    print("=" * 50)

    # Check Python version
    if not check_python_version():
        sys.exit(1)

    # Install dependencies if needed
    try:
        import streamlit
        print("✓ Core dependencies already installed")
    except ImportError:
        if not install_dependencies():
            sys.exit(1)

    # Download NLTK data
    if not download_nltk_data():
        print("⚠️  NLTK download failed, but continuing...")

    print("=" * 50)
    print("🚀 Starting application...")

    # Run the application
    run_application()

if __name__ == "__main__":
    main()

