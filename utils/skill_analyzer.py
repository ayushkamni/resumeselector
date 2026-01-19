"""
Skill Gap Analysis Module
Identifies missing skills and provides gap analysis between resume and job requirements
"""

from utils.nlp_processor import extract_skills, clean_text, tokenize_text
import streamlit as st

def analyze_skill_gaps(resume_skills: list, job_skills: list) -> list:
    """
    Analyze skill gaps between resume and job requirements

    Args:
        resume_skills (list): Skills found in resume
        job_skills (list): Skills required by job

    Returns:
        list: List of missing skills
    """
    try:
        if not job_skills:
            return []

        # Convert to sets for easier comparison
        resume_skills_set = set(skill.lower() for skill in resume_skills)
        job_skills_set = set(skill.lower() for skill in job_skills)

        # Find missing skills
        missing_skills = job_skills_set - resume_skills_set

        # Convert back to list and title case
        missing_skills_list = [skill.title() for skill in missing_skills]

        # Sort by priority (can be enhanced with skill importance scoring)
        missing_skills_list.sort()

        return missing_skills_list

    except Exception as e:
        st.warning(f"Skill gap analysis failed: {str(e)}")
        return []

def get_skill_overlap_analysis(resume_skills: list, job_skills: list) -> dict:
    """
    Provide detailed skill overlap analysis

    Args:
        resume_skills (list): Skills found in resume
        job_skills (list): Skills required by job

    Returns:
        dict: Detailed skill analysis
    """
    try:
        resume_skills_set = set(skill.lower() for skill in resume_skills)
        job_skills_set = set(skill.lower() for skill in job_skills)

        # Calculate overlaps
        matching_skills = resume_skills_set & job_skills_set
        missing_skills = job_skills_set - resume_skills_set
        extra_skills = resume_skills_set - job_skills_set

        # Calculate percentages
        total_job_skills = len(job_skills_set)
        match_percentage = (len(matching_skills) / total_job_skills * 100) if total_job_skills > 0 else 0

        return {
            'matching_skills': [skill.title() for skill in matching_skills],
            'missing_skills': [skill.title() for skill in missing_skills],
            'extra_skills': [skill.title() for skill in extra_skills],
            'match_percentage': round(match_percentage, 1),
            'total_job_skills': total_job_skills,
            'total_resume_skills': len(resume_skills_set)
        }

    except Exception as e:
        st.warning(f"Detailed skill analysis failed: {str(e)}")
        return {
            'matching_skills': [],
            'missing_skills': [],
            'extra_skills': [],
            'match_percentage': 0.0,
            'total_job_skills': 0,
            'total_resume_skills': 0
        }

def prioritize_missing_skills(missing_skills: list, job_description: str) -> list:
    """
    Prioritize missing skills based on their importance in job description

    Args:
        missing_skills (list): List of missing skills
        job_description (str): Job description text

    Returns:
        list: Prioritized list of missing skills
    """
    try:
        if not missing_skills or not job_description:
            return missing_skills

        job_text_lower = job_description.lower()

        # Count occurrences of each skill in job description
        skill_counts = {}
        for skill in missing_skills:
            skill_lower = skill.lower()
            count = job_text_lower.count(skill_lower)
            skill_counts[skill] = count

        # Sort by frequency (higher frequency = higher priority)
        prioritized_skills = sorted(skill_counts.items(), key=lambda x: x[1], reverse=True)
        return [skill for skill, count in prioritized_skills]

    except Exception as e:
        st.warning(f"Skill prioritization failed: {str(e)}")
        return missing_skills

def suggest_alternative_skills(missing_skill: str) -> list:
    """
    Suggest alternative or related skills for a missing skill

    Args:
        missing_skill (str): The missing skill

    Returns:
        list: List of alternative skills
    """
    # Skill mapping dictionary (can be expanded)
    skill_alternatives = {
        'python': ['java', 'r', 'scala', 'julia'],
        'java': ['kotlin', 'scala', 'c#', 'python'],
        'javascript': ['typescript', 'coffeescript', 'dart'],
        'sql': ['nosql', 'mongodb', 'postgresql', 'mysql'],
        'aws': ['azure', 'gcp', 'heroku', 'digitalocean'],
        'docker': ['kubernetes', 'podman', 'containerd'],
        'react': ['angular', 'vue', 'svelte', 'ember'],
        'machine learning': ['data science', 'ai', 'deep learning', 'statistics'],
        'tensorflow': ['pytorch', 'keras', 'scikit-learn', 'xgboost'],
        'git': ['svn', 'mercurial', 'perforce']
    }

    skill_lower = missing_skill.lower()

    if skill_lower in skill_alternatives:
        return [alt.title() for alt in skill_alternatives[skill_lower]]

    return []

def categorize_skills(skills: list) -> dict:
    """
    Categorize skills into technical and soft skills

    Args:
        skills (list): List of skills

    Returns:
        dict: Categorized skills
    """
    try:
        # Define skill categories
        technical_categories = {
            'Programming Languages': ['python', 'java', 'javascript', 'c++', 'c#', 'ruby', 'php', 'swift', 'kotlin', 'r', 'scala'],
            'Databases': ['sql', 'mysql', 'postgresql', 'mongodb', 'redis', 'elasticsearch', 'oracle'],
            'Web Technologies': ['html', 'css', 'react', 'angular', 'vue', 'node.js', 'django', 'flask', 'spring'],
            'Cloud Platforms': ['aws', 'azure', 'gcp', 'heroku', 'digitalocean'],
            'DevOps & Tools': ['docker', 'kubernetes', 'jenkins', 'git', 'linux', 'bash'],
            'AI/ML': ['machine learning', 'deep learning', 'nlp', 'computer vision', 'tensorflow', 'pytorch', 'scikit-learn']
        }

        soft_skills = ['communication', 'leadership', 'teamwork', 'problem solving', 'analytical', 'project management']

        categorized = {
            'Technical Skills': {},
            'Soft Skills': []
        }

        skills_lower = [skill.lower() for skill in skills]

        # Categorize technical skills
        for category, category_skills in technical_categories.items():
            matching_skills = [skill.title() for skill in skills
                             if skill.lower() in category_skills]
            if matching_skills:
                categorized['Technical Skills'][category] = matching_skills

        # Find soft skills
        for skill in skills:
            if skill.lower() in soft_skills:
                categorized['Soft Skills'].append(skill)

        return categorized

    except Exception as e:
        st.warning(f"Skill categorization failed: {str(e)}")
        return {'Technical Skills': {}, 'Soft Skills': []}

