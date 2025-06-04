# job_parser.py
import re

COMMON_SKILLS = ['python', 'machine learning', 'deep learning', 'pytorch', 'tensorflow', 'nlp', 'data science']
COMMON_CERTS = ['aws certified', 'azure', 'gcp', 'tensorflow certification']
DEGREES = ['bachelor', 'master', 'phd', 'b.tech', 'm.tech']

def parse_job_description(job_text: str):
    job_text_lower = job_text.lower()

    skills = [skill for skill in COMMON_SKILLS if skill in job_text_lower]

    # Estimate required years of experience
    match = re.search(r'(\\d+)\\+?\\s+years?', job_text_lower)
    required_experience = float(match.group(1)) if match else 0.0

    education = ''
    for degree in DEGREES:
        if degree in job_text_lower:
            education = degree
            break

    certifications = [cert for cert in COMMON_CERTS if cert in job_text_lower]

    return {
        'skills': skills,
        'required_experience': required_experience,
        'education': education,
        'certifications': certifications
    }
