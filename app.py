"""
Smart AI Resume Analyzer
A comprehensive resume analysis tool that matches resumes with job descriptions
using NLP techniques and provides actionable insights.

Author: AI Resume Analyzer Team
"""

import streamlit as st
import pandas as pd
import numpy as np
from utils.text_extractor import extract_text_from_file
from utils.nlp_processor import clean_text, extract_skills
from utils.similarity_scorer import calculate_similarity_score
from utils.skill_analyzer import analyze_skill_gaps
from utils.improvement_suggester import generate_improvements
import plotly.graph_objects as go
import plotly.express as px
from io import StringIO

# Configure page
st.set_page_config(
    page_title="Smart AI Resume Analyzer",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better UI
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .score-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #1f77b4;
        margin: 1rem 0;
    }
    .skill-gap {
        background-color: #fff3cd;
        padding: 1rem;
        border-radius: 5px;
        border-left: 5px solid #ffc107;
        margin: 0.5rem 0;
    }
    .improvement {
        background-color: #d1ecf1;
        padding: 1rem;
        border-radius: 5px;
        border-left: 5px solid #17a2b8;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

def main():
    """Main application function"""
    st.markdown('<h1 class="main-header">🎯 Smart AI Resume Analyzer</h1>', unsafe_allow_html=True)
    st.markdown("---")

    # Sidebar for inputs
    with st.sidebar:
        st.header("📤 Input Section")

        # Resume upload
        st.subheader("Upload Resume")
        uploaded_file = st.file_uploader(
            "Choose a resume file (PDF/DOCX)",
            type=['pdf', 'docx'],
            help="Upload your resume in PDF or DOCX format"
        )

        # Job description input
        st.subheader("Job Description")
        job_description = st.text_area(
            "Paste the job description here",
            height=200,
            help="Copy and paste the complete job description"
        )

        # Analysis button
        analyze_button = st.button(
            "🔍 Analyze Resume",
            type="primary",
            use_container_width=True
        )

    # Main content area
    if not uploaded_file or not job_description or not analyze_button:
        # Welcome screen
        col1, col2 = st.columns([2, 1])

        with col1:
            st.markdown("## Welcome to Smart AI Resume Analyzer! 🚀")
            st.markdown("""
            This AI-powered tool helps you:
            - **Match your resume** with job descriptions
            - **Identify skill gaps** and missing keywords
            - **Get personalized improvement suggestions**
            - **Boost your ATS compatibility score**

            ### How it works:
            1. 📤 Upload your resume (PDF or DOCX)
            2. 📝 Paste the job description
            3. 🔍 Click "Analyze Resume" to get insights
            4. 📊 Review your match score and recommendations
            """)

        with col2:
            st.markdown("### 📊 Sample Analysis")
            # Placeholder for sample results
            st.info("Upload a resume and job description to see analysis results here!")

    else:
        # Perform analysis
        with st.spinner("Analyzing your resume... 🤖"):
            try:
                # Extract text from resume
                resume_text = extract_text_from_file(uploaded_file)

                if not resume_text.strip():
                    st.error("❌ Could not extract text from the resume. Please check the file format.")
                    return

                # Process texts
                cleaned_resume = clean_text(resume_text)
                cleaned_job = clean_text(job_description)

                # Calculate similarity score
                similarity_score = calculate_similarity_score(cleaned_resume, cleaned_job)

                # Extract skills
                resume_skills = extract_skills(cleaned_resume)
                job_skills = extract_skills(cleaned_job)

                # Analyze skill gaps
                skill_gaps = analyze_skill_gaps(resume_skills, job_skills)

                # Generate improvements
                improvements = generate_improvements(cleaned_resume, cleaned_job, skill_gaps)

                # Display results
                display_results(
                    similarity_score,
                    resume_skills,
                    job_skills,
                    skill_gaps,
                    improvements,
                    resume_text
                )

            except Exception as e:
                st.error(f"❌ An error occurred during analysis: {str(e)}")
                st.info("💡 Try uploading a different file or check if the job description is properly formatted.")

def display_results(score, resume_skills, job_skills, gaps, improvements, resume_text):
    """Display analysis results in organized sections"""

    # Overall Score Section
    st.markdown("## 📊 Overall Match Score")
    col1, col2, col3 = st.columns([2, 1, 1])

    with col1:
        # Score gauge
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=score,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': "Match Score"},
            gauge={
                'axis': {'range': [0, 100]},
                'bar': {'color': "#1f77b4"},
                'steps': [
                    {'range': [0, 40], 'color': "#ff6b6b"},
                    {'range': [40, 70], 'color': "#ffd93d"},
                    {'range': [70, 100], 'color': "#6bcf7f"}
                ]
            }
        ))
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown('<div class="score-card">', unsafe_allow_html=True)
        st.markdown(f"**Score: {score:.1f}%**")
        if score >= 80:
            st.success("🎉 Excellent match!")
        elif score >= 60:
            st.warning("👍 Good match")
        else:
            st.error("⚠️ Needs improvement")
        st.markdown('</div>', unsafe_allow_html=True)

    with col3:
        st.markdown('<div class="score-card">', unsafe_allow_html=True)
        st.markdown("**Resume Length**")
        st.markdown(f"{len(resume_text.split())} words")
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("---")

    # Skills Analysis Section
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("## 🎯 Your Skills")
        if resume_skills:
            skills_df = pd.DataFrame({
                'Skill': resume_skills[:10],  # Show top 10
                'Source': ['Resume'] * len(resume_skills[:10])
            })
            st.dataframe(skills_df, use_container_width=True)
        else:
            st.info("No skills detected in resume")

    with col2:
        st.markdown("## 💼 Required Skills")
        if job_skills:
            skills_df = pd.DataFrame({
                'Skill': job_skills[:10],  # Show top 10
                'Source': ['Job Description'] * len(job_skills[:10])
            })
            st.dataframe(skills_df, use_container_width=True)
        else:
            st.info("No skills detected in job description")

    # Skills comparison visualization
    if resume_skills and job_skills:
        st.markdown("### Skills Overlap Analysis")
        common_skills = set(resume_skills) & set(job_skills)
        missing_skills = set(job_skills) - set(resume_skills)

        fig = go.Figure()
        fig.add_trace(go.Bar(
            name='Matching Skills',
            x=['Skills'],
            y=[len(common_skills)],
            marker_color='#6bcf7f'
        ))
        fig.add_trace(go.Bar(
            name='Missing Skills',
            x=['Skills'],
            y=[len(missing_skills)],
            marker_color='#ff6b6b'
        ))
        fig.update_layout(barmode='stack', title="Skills Analysis")
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    # Skill Gaps Section
    st.markdown("## ⚠️ Skill Gaps")
    if gaps:
        for gap in gaps[:5]:  # Show top 5 gaps
            st.markdown(f'<div class="skill-gap">🔸 **{gap}** - Consider adding this to your resume</div>',
                       unsafe_allow_html=True)
    else:
        st.success("🎉 No major skill gaps detected!")

    st.markdown("---")

    # Improvement Suggestions
    st.markdown("## 💡 Improvement Suggestions")
    if improvements:
        for i, improvement in enumerate(improvements[:5], 1):  # Show top 5
            st.markdown(f'<div class="improvement">{i}. {improvement}</div>',
                       unsafe_allow_html=True)
    else:
        st.info("No specific improvements suggested - your resume looks good!")

    # Download report option
    st.markdown("---")
    st.markdown("## 📥 Download Analysis Report")

    report_data = f"""
    Resume Analysis Report
    =====================

    Match Score: {score:.1f}%

    Resume Skills ({len(resume_skills)}):
    {', '.join(resume_skills[:10])}

    Required Skills ({len(job_skills)}):
    {', '.join(job_skills[:10])}

    Skill Gaps ({len(gaps)}):
    {', '.join(gaps[:5])}

    Generated on: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}
    """

    st.download_button(
        label="📄 Download Report",
        data=report_data,
        file_name="resume_analysis_report.txt",
        mime="text/plain"
    )

if __name__ == "__main__":
    main()

