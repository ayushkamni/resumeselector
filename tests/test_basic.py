"""
Basic tests for Smart AI Resume Analyzer
Run with: python -m pytest tests/test_basic.py
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.nlp_processor import clean_text, extract_skills
from utils.similarity_scorer import calculate_similarity_score
from utils.skill_analyzer import analyze_skill_gaps
import pytest

def test_text_cleaning():
    """Test text cleaning functionality"""
    dirty_text = "Hello!!! This is a TEST text with URLs: https://example.com and emails: test@email.com"
    cleaned = clean_text(dirty_text)

    assert "!!!" not in cleaned
    assert "https://" not in cleaned
    assert "@" not in cleaned
    assert cleaned == cleaned.lower()  # Should be lowercase
    print("✓ Text cleaning test passed")

def test_skill_extraction():
    """Test skill extraction from text"""
    text = "I am proficient in Python, JavaScript, and machine learning using TensorFlow."
    skills = extract_skills(text)

    assert "Python" in skills or "python" in [s.lower() for s in skills]
    assert len(skills) > 0
    print("✓ Skill extraction test passed")

def test_similarity_scoring():
    """Test similarity scoring between texts"""
    text1 = "Python developer with Django experience"
    text2 = "Java developer with Spring experience"
    text3 = "Python developer with Django and Flask experience"

    score1 = calculate_similarity_score(text1, text2)
    score2 = calculate_similarity_score(text1, text3)

    assert score2 > score1  # text1 should be more similar to text3
    assert 0 <= score1 <= 100
    assert 0 <= score2 <= 100
    print("✓ Similarity scoring test passed")

def test_skill_gap_analysis():
    """Test skill gap analysis"""
    resume_skills = ["Python", "Django", "SQL"]
    job_skills = ["Python", "Django", "SQL", "AWS", "Docker"]

    gaps = analyze_skill_gaps(resume_skills, job_skills)

    assert "AWS" in gaps or "aws" in [g.lower() for g in gaps]
    assert "Docker" in gaps or "docker" in [g.lower() for g in gaps]
    assert "Python" not in gaps
    print("✓ Skill gap analysis test passed")

def test_empty_inputs():
    """Test handling of empty inputs"""
    # Test empty text similarity
    score = calculate_similarity_score("", "hello world")
    assert score == 0.0

    # Test empty skill extraction
    skills = extract_skills("")
    assert skills == []

    # Test empty skill gaps
    gaps = analyze_skill_gaps([], ["python"])
    assert len(gaps) > 0

    print("✓ Empty input handling test passed")

if __name__ == "__main__":
    print("Running basic tests for Smart AI Resume Analyzer...")
    print("=" * 50)

    try:
        test_text_cleaning()
        test_skill_extraction()
        test_similarity_scoring()
        test_skill_gap_analysis()
        test_empty_inputs()

        print("=" * 50)
        print("🎉 All tests passed successfully!")

    except Exception as e:
        print(f"❌ Test failed: {str(e)}")
        sys.exit(1)

