"""
Improvement Suggestions Module
Generates actionable improvement suggestions based on resume analysis
"""

from utils.nlp_processor import get_text_statistics, clean_text
from utils.similarity_scorer import get_similarity_details, get_readability_score
import streamlit as st
import re

def generate_improvements(resume_text: str, job_text: str, skill_gaps: list) -> list:
    """
    Generate comprehensive improvement suggestions

    Args:
        resume_text (str): Original resume text
        job_text (str): Job description text
        skill_gaps (list): List of missing skills

    Returns:
        list: List of improvement suggestions
    """
    suggestions = []

    try:
        # Get similarity details
        similarity_details = get_similarity_details(resume_text, job_text)

        # Get text statistics
        resume_stats = get_text_statistics(resume_text)
        readability = get_readability_score(resume_text)

        # 1. Skill-based suggestions
        skill_suggestions = generate_skill_suggestions(skill_gaps, job_text)
        suggestions.extend(skill_suggestions)

        # 2. Content-based suggestions
        content_suggestions = generate_content_suggestions(
            resume_text, job_text, similarity_details
        )
        suggestions.extend(content_suggestions)

        # 3. Structure-based suggestions
        structure_suggestions = generate_structure_suggestions(resume_stats, readability)
        suggestions.extend(structure_suggestions)

        # 4. Keyword optimization suggestions
        keyword_suggestions = generate_keyword_suggestions(similarity_details)
        suggestions.extend(keyword_suggestions)

        # Remove duplicates and limit to top suggestions
        suggestions = list(set(suggestions))[:10]

        return suggestions

    except Exception as e:
        st.warning(f"Improvement generation failed: {str(e)}")
        return ["Review and update your resume with job-specific keywords and skills"]

def generate_skill_suggestions(skill_gaps: list, job_text: str) -> list:
    """Generate skill-related improvement suggestions"""
    suggestions = []

    if skill_gaps:
        suggestions.append(f"Add these missing skills to your resume: {', '.join(skill_gaps[:3])}")

        # Check if skills are mentioned in job description
        for skill in skill_gaps[:2]:
            skill_lower = skill.lower()
            if skill_lower in job_text.lower():
                suggestions.append(f"Highlight your experience with {skill} more prominently")

    if len(skill_gaps) > 5:
        suggestions.append("Consider gaining proficiency in some of the missing skills through online courses")

    return suggestions

def generate_content_suggestions(resume_text: str, job_text: str, similarity_details: dict) -> list:
    """Generate content-related improvement suggestions"""
    suggestions = []

    # Check for job-specific keywords
    job_unique_terms = similarity_details.get('job_unique_terms', [])
    if job_unique_terms:
        top_keywords = job_unique_terms[:5]
        suggestions.append(f"Incorporate these job-specific keywords: {', '.join(top_keywords)}")

    # Check resume length
    word_count = len(resume_text.split())
    if word_count < 200:
        suggestions.append("Your resume seems short. Consider adding more relevant experience and achievements")
    elif word_count > 800:
        suggestions.append("Your resume is quite long. Consider condensing it to focus on the most relevant information")

    # Check for quantifiable achievements
    achievement_indicators = ['increased', 'improved', 'reduced', 'achieved', 'delivered', 'managed', 'led']
    has_achievements = any(indicator in resume_text.lower() for indicator in achievement_indicators)
    if not has_achievements:
        suggestions.append("Add quantifiable achievements and metrics to demonstrate your impact")

    # Check for action verbs
    action_verbs = ['developed', 'created', 'implemented', 'designed', 'managed', 'led', 'optimized']
    has_action_verbs = any(verb in resume_text.lower() for verb in action_verbs)
    if not has_action_verbs:
        suggestions.append("Use strong action verbs to describe your accomplishments")

    return suggestions

def generate_structure_suggestions(stats: dict, readability: dict) -> list:
    """Generate structure and formatting suggestions"""
    suggestions = []

    # Check sentence complexity
    avg_words = readability.get('avg_words_per_sentence', 0)
    if avg_words > 25:
        suggestions.append("Shorten complex sentences for better readability")

    # Check readability score
    flesch_score = readability.get('flesch_reading_ease', 0)
    if flesch_score < 40:
        suggestions.append("Simplify language and technical jargon for better readability")

    # Check word variety
    total_words = stats.get('total_words', 1)
    if total_words > 0:
        unique_words_ratio = stats.get('unique_words', 0) / total_words
        if unique_words_ratio < 0.3:
            suggestions.append("Use more varied vocabulary to make your resume more engaging")

    return suggestions

def generate_keyword_suggestions(similarity_details: dict) -> list:
    """Generate keyword optimization suggestions"""
    suggestions = []

    common_terms = similarity_details.get('common_terms', [])
    resume_unique = similarity_details.get('resume_unique_terms', [])
    job_unique = similarity_details.get('job_unique_terms', [])

    if len(common_terms) < 5:
        suggestions.append("Increase keyword alignment by incorporating more job-relevant terms")

    if job_unique:
        suggestions.append("Consider adding these industry-specific terms that appear in the job description")

    if resume_unique and len(resume_unique) > len(common_terms):
        suggestions.append("Some of your unique skills might not be relevant - focus on job requirements")

    return suggestions

def get_resume_score_interpretation(score: float) -> str:
    """
    Provide interpretation of resume match score

    Args:
        score (float): Match score percentage

    Returns:
        str: Score interpretation
    """
    if score >= 80:
        return "Excellent match! Your resume is well-aligned with the job requirements."
    elif score >= 70:
        return "Good match. Your resume covers most job requirements with minor gaps."
    elif score >= 60:
        return "Fair match. Consider addressing skill gaps and adding relevant keywords."
    elif score >= 40:
        return "Below average match. Significant improvements needed in skills and keywords."
    else:
        return "Poor match. Major revisions required to align with job requirements."

def generate_actionable_tips(score: float, skill_gaps: list) -> list:
    """
    Generate specific actionable tips based on score and gaps

    Args:
        score (float): Match score
        skill_gaps (list): List of missing skills

    Returns:
        list: Actionable tips
    """
    tips = []

    if score < 60:
        tips.extend([
            "Take online courses to acquire missing skills",
            "Network with professionals in the target industry",
            "Update your resume with recent relevant projects"
        ])

    if skill_gaps:
        tips.append(f"Focus on gaining experience in: {', '.join(skill_gaps[:3])}")

    if score >= 60:
        tips.extend([
            "Customize your resume further for this specific job",
            "Prepare specific examples of your work for interviews",
            "Highlight transferable skills from related experience"
        ])

    return tips[:5]  # Limit to top 5 tips

def suggest_certifications(skill_gaps: list) -> list:
    """
    Suggest relevant certifications based on skill gaps

    Args:
        skill_gaps (list): List of missing skills

    Returns:
        list: Suggested certifications
    """
    certification_map = {
        'python': ['Python Institute PCAP', 'Google IT Automation with Python'],
        'aws': ['AWS Certified Solutions Architect', 'AWS Certified Developer'],
        'azure': ['Microsoft Azure Fundamentals', 'Microsoft Azure Administrator'],
        'machine learning': ['Google Machine Learning Crash Course', 'Coursera ML Specialization'],
        'docker': ['Docker Certified Associate', 'Kubernetes Certification'],
        'sql': ['Oracle SQL Certification', 'Microsoft SQL Server Certification']
    }

    suggestions = []
    for skill in skill_gaps[:3]:  # Top 3 gaps
        skill_lower = skill.lower()
        if skill_lower in certification_map:
            certs = certification_map[skill_lower][:2]  # Max 2 per skill
            suggestions.extend([f"Consider {cert} certification" for cert in certs])

    return suggestions[:4]  # Max 4 certification suggestions

