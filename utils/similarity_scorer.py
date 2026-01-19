"""
Similarity Scoring Module
Calculates resume-job match scores using TF-IDF and cosine similarity
"""

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from utils.nlp_processor import clean_text, remove_stopwords, tokenize_text, extract_skills
import streamlit as st

def calculate_jaccard_similarity(text1: str, text2: str) -> float:
    """
    Calculate Jaccard similarity as fallback when TF-IDF fails

    Args:
        text1 (str): First text
        text2 (str): Second text

    Returns:
        float: Similarity score as percentage (0-100)
    """
    try:
        # Tokenize and create sets
        tokens1 = set(tokenize_text(text1.lower()))
        tokens2 = set(tokenize_text(text2.lower()))

        # Remove stopwords
        tokens1 = set(remove_stopwords(list(tokens1)))
        tokens2 = set(remove_stopwords(list(tokens2)))

        # Calculate Jaccard similarity
        intersection = len(tokens1 & tokens2)
        union = len(tokens1 | tokens2)

        if union == 0:
            return 0.0

        jaccard_score = intersection / union
        return round(jaccard_score * 100, 1)

    except Exception as e:
        st.warning(f"Jaccard similarity calculation failed: {str(e)}")
        return 0.0

def calculate_tfidf_similarity(resume_text: str, job_text: str) -> float:
    """
    Calculate TF-IDF based text similarity (40% of total hybrid score)
    Focuses on overall text similarity and context

    Args:
        resume_text (str): Cleaned resume text
        job_text (str): Cleaned job description text

    Returns:
        float: TF-IDF similarity score (0-100)
    """
    try:
        # Create TF-IDF vectorizer optimized for resume-job matching
        vectorizer = TfidfVectorizer(
            max_features=1000,
            stop_words='english',
            ngram_range=(1, 2),  # Include unigrams and bigrams
            min_df=1,
            max_df=1.0,  # Allow terms that appear in all documents
            sublinear_tf=True,  # Better handling of term frequency
            use_idf=True,
            smooth_idf=True,
            norm='l2'  # L2 normalization for cosine similarity
        )

        documents = [resume_text, job_text]
        tfidf_matrix = vectorizer.fit_transform(documents)

        # Check if we have enough features
        if tfidf_matrix.shape[1] < 2:
            return calculate_jaccard_similarity(resume_text, job_text)

        # Calculate cosine similarity
        similarity_matrix = cosine_similarity(tfidf_matrix)
        similarity_score = similarity_matrix[0][1]

        # Ensure score is between 0 and 1
        similarity_score = max(0.0, min(1.0, similarity_score))

        return similarity_score * 100

    except Exception as e:
        return calculate_jaccard_similarity(resume_text, job_text)

def calculate_skill_similarity(resume_text: str, job_text: str) -> float:
    """
    Calculate skill-based similarity (60% of total hybrid score)
    This is the MOST IMPORTANT factor for resume-job matching

    Args:
        resume_text (str): Cleaned resume text
        job_text (str): Cleaned job description text

    Returns:
        float: Skill similarity score (0-100)
    """
    try:
        # Extract skills from both texts using existing skill extraction
        resume_skills = set(extract_skills(resume_text))
        job_skills = set(extract_skills(job_text))

        if not resume_skills and not job_skills:
            return 0.0

        if not resume_skills or not job_skills:
            return 0.0

        # Calculate skill overlap using Jaccard similarity
        common_skills = resume_skills & job_skills
        total_unique_skills = resume_skills | job_skills

        # Jaccard similarity: intersection / union
        skill_similarity = len(common_skills) / len(total_unique_skills) if total_unique_skills else 0

        # Boost score for strong skill matches (very important for job matching)
        if len(common_skills) >= 3:  # If 3+ skills match, boost the score
            skill_similarity = min(1.0, skill_similarity * 1.2)  # 20% boost for strong matches

        return skill_similarity * 100

    except Exception as e:
        st.warning(f"Skill similarity calculation failed: {str(e)}")
        return 0.0

def calculate_similarity_score(resume_text: str, job_text: str) -> float:
    """
    Calculate similarity score between resume and job description

    Args:
        resume_text (str): Cleaned resume text
        job_text (str): Cleaned job description text

    Returns:
        float: Similarity score as percentage (0-100)
    """
    try:
        if not resume_text or not job_text:
            return 0.0

        # HYBRID SCORING APPROACH: 40% TF-IDF + 60% Skill Overlap
        # This is crucial for resume-job matching accuracy

        # Calculate TF-IDF similarity (40% weight) - for overall text/context matching
        tfidf_score = calculate_tfidf_similarity(resume_text, job_text)

        # Calculate skill-based similarity (60% weight) - most important for job matching
        skill_score = calculate_skill_similarity(resume_text, job_text)

        # Combine scores: TF-IDF gives context, skills give technical relevance
        hybrid_score = (tfidf_score * 0.4) + (skill_score * 0.6)

        # Ensure score is between 0 and 100
        hybrid_score = max(0.0, min(100.0, hybrid_score))

        return round(hybrid_score, 1)

    except Exception as e:
        st.error(f"Hybrid similarity calculation failed: {str(e)}")
        return 0.0

def get_similarity_details(resume_text: str, job_text: str) -> dict:
    """
    Get detailed similarity analysis

    Args:
        resume_text (str): Cleaned resume text
        job_text (str): Cleaned job description text

    Returns:
        dict: Detailed similarity information
    """
    try:
        if not resume_text or not job_text:
            return {
                'overall_score': 0.0,
                'common_terms': [],
                'resume_unique_terms': [],
                'job_unique_terms': []
            }

        # Create TF-IDF vectorizer
        vectorizer = TfidfVectorizer(
            max_features=1000,
            stop_words='english',
            ngram_range=(1, 2)
        )

        documents = [resume_text, job_text]
        tfidf_matrix = vectorizer.fit_transform(documents)

        # Get feature names (terms)
        feature_names = vectorizer.get_feature_names_out()

        # Get TF-IDF scores for each document
        resume_tfidf = tfidf_matrix[0].toarray()[0]
        job_tfidf = tfidf_matrix[1].toarray()[0]

        # Find common terms (terms that appear in both with significant scores)
        common_terms = []
        resume_unique = []
        job_unique = []

        threshold = 0.1  # Minimum TF-IDF score to consider

        for i, term in enumerate(feature_names):
            resume_score = resume_tfidf[i]
            job_score = job_tfidf[i]

            if resume_score > threshold and job_score > threshold:
                common_terms.append((term, (resume_score + job_score) / 2))
            elif resume_score > threshold:
                resume_unique.append((term, resume_score))
            elif job_score > threshold:
                job_unique.append((term, job_score))

        # Sort by TF-IDF score
        common_terms.sort(key=lambda x: x[1], reverse=True)
        resume_unique.sort(key=lambda x: x[1], reverse=True)
        job_unique.sort(key=lambda x: x[1], reverse=True)

        # Calculate overall score
        overall_score = calculate_similarity_score(resume_text, job_text)

        return {
            'overall_score': overall_score,
            'common_terms': [term for term, score in common_terms[:10]],  # Top 10
            'resume_unique_terms': [term for term, score in resume_unique[:10]],  # Top 10
            'job_unique_terms': [term for term, score in job_unique[:10]]  # Top 10
        }

    except Exception as e:
        st.error(f"Detailed similarity analysis failed: {str(e)}")
        return {
            'overall_score': 0.0,
            'common_terms': [],
            'resume_unique_terms': [],
            'job_unique_terms': []
        }

def calculate_keyword_match_score(resume_text: str, job_keywords: list) -> float:
    """
    Calculate how well resume matches specific job keywords

    Args:
        resume_text (str): Cleaned resume text
        job_keywords (list): List of important job keywords

    Returns:
        float: Keyword match score as percentage
    """
    try:
        if not resume_text or not job_keywords:
            return 0.0

        resume_tokens = tokenize_text(resume_text.lower())
        resume_tokens = remove_stopwords(resume_tokens)
        resume_token_set = set(resume_tokens)

        matched_keywords = 0
        for keyword in job_keywords:
            keyword_lower = keyword.lower()
            # Check if keyword or its variations exist in resume
            if keyword_lower in resume_token_set or keyword_lower in resume_text.lower():
                matched_keywords += 1

        match_score = (matched_keywords / len(job_keywords)) * 100
        return round(match_score, 1)

    except Exception as e:
        st.warning(f"Keyword match calculation failed: {str(e)}")
        return 0.0

def get_readability_score(text: str) -> dict:
    """
    Calculate basic readability metrics

    Args:
        text (str): Input text

    Returns:
        dict: Readability metrics
    """
    try:
        sentences = text.split('.')
        words = text.split()
        total_sentences = len([s for s in sentences if s.strip()])
        total_words = len(words)
        total_syllables = sum(count_syllables(word) for word in words)

        # Flesch Reading Ease Score
        if total_sentences > 0 and total_words > 0:
            flesch_score = 206.835 - 1.015 * (total_words / total_sentences) - 84.6 * (total_syllables / total_words)
            flesch_score = max(0, min(100, flesch_score))  # Clamp between 0-100
        else:
            flesch_score = 0

        return {
            'flesch_reading_ease': round(flesch_score, 1),
            'total_sentences': total_sentences,
            'total_words': total_words,
            'avg_words_per_sentence': round(total_words / total_sentences, 1) if total_sentences > 0 else 0
        }

    except Exception as e:
        return {
            'flesch_reading_ease': 0,
            'total_sentences': 0,
            'total_words': 0,
            'avg_words_per_sentence': 0
        }

def count_syllables(word: str) -> int:
    """
    Count syllables in a word (basic implementation)

    Args:
        word (str): Input word

    Returns:
        int: Number of syllables
    """
    word = word.lower()
    count = 0
    vowels = "aeiouy"

    if word[0] in vowels:
        count += 1

    for i in range(1, len(word)):
        if word[i] in vowels and word[i - 1] not in vowels:
            count += 1

    if word.endswith("e"):
        count -= 1

    if count == 0:
        count += 1

    return count

