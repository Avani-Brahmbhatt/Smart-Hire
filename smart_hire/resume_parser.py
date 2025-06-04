# resume_parser.py
import re

COMMON_SKILLS = ['python', 'machine learning', 'deep learning', 'pytorch', 'tensorflow', 'nlp', 'data science']
COMMON_CERTS = ['aws certified', 'azure', 'gcp', 'tensorflow certification']
DEGREES = ['bachelor', 'master', 'phd', 'b.tech', 'm.tech']

def parse_resume(resume_text: str):
    resume_text_lower = resume_text.lower()

    skills = [skill for skill in COMMON_SKILLS if skill in resume_text_lower]

    # Extract years of experience using regex
    match = re.search(r'(\\d+)\\+?\\s+years?', resume_text_lower)
    years_experience = float(match.group(1)) if match else 0.0

    education = ''
    for degree in DEGREES:
        if degree in resume_text_lower:
            education = degree
            break

    certifications = [cert for cert in COMMON_CERTS if cert in resume_text_lower]

    return {
        'skills': skills,
        'years_experience': years_experience,
        'education': education,
        'certifications': certifications
    }
