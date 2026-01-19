"""
Setup script to download NLTK data for deployment
Run this during build process to ensure NLTK resources are available
"""

import nltk
import ssl

# Disable SSL verification for NLTK downloads (sometimes needed in cloud environments)
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

def download_nltk_data():
    """Download required NLTK data packages"""

    packages = [
        'punkt',
        'punkt_tab',
        'stopwords',
        'wordnet',
        'averaged_perceptron_tagger',
        'omw-1.4'
    ]

    print("Downloading NLTK data packages...")

    for package in packages:
        try:
            print(f"Downloading {package}...")
            nltk.download(package, quiet=True)
            print(f"✓ {package} downloaded successfully")
        except Exception as e:
            print(f"⚠️ Failed to download {package}: {e}")
            continue

    print("NLTK data download complete!")

if __name__ == "__main__":
    download_nltk_data()
