"""
NLP Processing Module
Handles text cleaning, preprocessing, and skill extraction using NLTK and spaCy
"""

import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import WordNetLemmatizer
import string
import streamlit as st

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    try:
        nltk.download('punkt')
    except Exception:
        # For newer NLTK versions
        nltk.download('punkt_tab')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')

# Predefined skill keywords (can be expanded)
TECHNICAL_SKILLS = {
    'python', 'java', 'javascript', 'c++', 'c#', 'ruby', 'php', 'swift', 'kotlin',
    'sql', 'mysql', 'postgresql', 'mongodb', 'redis', 'elasticsearch',
    'html', 'css', 'react', 'angular', 'vue', 'node.js', 'django', 'flask',
    'aws', 'azure', 'gcp', 'docker', 'kubernetes', 'jenkins', 'git',
    'machine learning', 'deep learning', 'nlp', 'computer vision', 'tensorflow',
    'pytorch', 'scikit-learn', 'pandas', 'numpy', 'matplotlib', 'seaborn',
    'linux', 'windows', 'macos', 'api', 'rest', 'graphql', 'microservices'
}

SOFT_SKILLS = {
    'communication', 'leadership', 'teamwork', 'problem solving', 'analytical',
    'project management', 'time management', 'creativity', 'adaptability',
    'critical thinking', 'collaboration', 'mentoring', 'presentation'
}

def clean_text(text: str) -> str:
    """
    Clean and preprocess text for analysis

    Args:
        text (str): Raw text input

    Returns:
        str: Cleaned and processed text
    """
    if not text:
        return ""

    try:
        # Convert to lowercase
        text = text.lower()

        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)

        # Remove email addresses
        text = re.sub(r'\S+@\S+', '', text)

        # Remove phone numbers
        text = re.sub(r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b', '', text)

        # Remove special characters but keep spaces and alphanumeric
        text = re.sub(r'[^\w\s]', ' ', text)

        # Remove extra whitespace
        text = ' '.join(text.split())

        return text.strip()

    except Exception as e:
        st.warning(f"Text cleaning failed: {str(e)}")
        return text

def tokenize_text(text: str) -> list:
    """
    Tokenize text into words

    Args:
        text (str): Input text

    Returns:
        list: List of tokens
    """
    try:
        tokens = word_tokenize(text)
        return tokens
    except LookupError as e:
        # If punkt tokenizer is not available, try alternative tokenization
        st.warning(f"NLTK tokenizer not available: {str(e)}. Using fallback tokenization.")
        # Simple fallback tokenization
        import re
        tokens = re.findall(r'\b\w+\b', text.lower())
        return tokens
    except Exception as e:
        st.warning(f"Tokenization failed: {str(e)}")
        return text.split()

def remove_stopwords(tokens: list) -> list:
    """
    Remove stopwords from token list

    Args:
        tokens (list): List of tokens

    Returns:
        list: Filtered tokens without stopwords
    """
    try:
        stop_words = set(stopwords.words('english'))
        # Add custom stopwords
        custom_stopwords = {'also', 'would', 'could', 'should', 'may', 'might', 'must'}
        stop_words.update(custom_stopwords)

        filtered_tokens = [word for word in tokens if word.lower() not in stop_words]
        return filtered_tokens
    except Exception as e:
        st.warning(f"Stopword removal failed: {str(e)}")
        return tokens

def lemmatize_tokens(tokens: list) -> list:
    """
    Lemmatize tokens to their base form

    Args:
        tokens (list): List of tokens

    Returns:
        list: Lemmatized tokens
    """
    try:
        lemmatizer = WordNetLemmatizer()
        lemmatized = [lemmatizer.lemmatize(token) for token in tokens]
        return lemmatized
    except Exception as e:
        st.warning(f"Lemmatization failed: {str(e)}")
        return tokens

def extract_skills(text: str) -> list:
    """
    Extract skills from text using predefined skill sets

    Args:
        text (str): Input text

    Returns:
        list: List of extracted skills
    """
    if not text:
        return []

    try:
        # Clean and tokenize text
        cleaned_text = clean_text(text)
        tokens = tokenize_text(cleaned_text)

        # Convert to set for faster lookup
        token_set = set(tokens)

        # Find matching skills
        found_skills = []

        # Check for exact matches
        for skill in TECHNICAL_SKILLS | SOFT_SKILLS:
            if skill in cleaned_text.lower():
                found_skills.append(skill.title())

        # Also check for partial matches in tokens
        for token in tokens:
            token_lower = token.lower()
            if token_lower in TECHNICAL_SKILLS or token_lower in SOFT_SKILLS:
                if token_lower.title() not in found_skills:
                    found_skills.append(token_lower.title())

        # Remove duplicates and sort
        found_skills = list(set(found_skills))
        found_skills.sort()

        return found_skills

    except Exception as e:
        st.warning(f"Skill extraction failed: {str(e)}")
        return []

def get_text_statistics(text: str) -> dict:
    """
    Get basic statistics about the text

    Args:
        text (str): Input text

    Returns:
        dict: Text statistics
    """
    try:
        sentences = sent_tokenize(text)
        words = tokenize_text(text)
        cleaned_words = remove_stopwords(words)

        return {
            'total_sentences': len(sentences),
            'total_words': len(words),
            'unique_words': len(set(words)),
            'cleaned_words': len(cleaned_words),
            'avg_words_per_sentence': len(words) / len(sentences) if sentences else 0
        }
    except Exception as e:
        return {
            'total_sentences': 0,
            'total_words': 0,
            'unique_words': 0,
            'cleaned_words': 0,
            'avg_words_per_sentence': 0
        }

