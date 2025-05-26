# utils.py
import os
import json
import re
from typing import List, Dict, Any
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def ensure_directory_exists(directory: str):
    """Create directory if it doesn't exist"""
    if not os.path.exists(directory):
        os.makedirs(directory)
        logger.info(f"Created directory: {directory}")

def extract_email_from_text(text: str) -> str:
    """Extract email address from text"""
    email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
    emails = re.findall(email_pattern, text)
    return emails[0] if emails else ""

def extract_phone_from_text(text: str) -> str:
    """Extract phone number from text"""
    phone_pattern = r'(\+\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}'
    phones = re.findall(phone_pattern, text)
    return phones[0] if phones else ""

def extract_name_from_filename(filename: str) -> str:
    """Extract candidate name from filename"""
    name = os.path.splitext(filename)[0]
    name = re.sub(r'[_-]', ' ', name)
    return name.title()

def safe_json_loads(json_str: str, default=None):
    """Safely load JSON string"""
    try:
        return json.loads(json_str) if json_str else default or {}
    except json.JSONDecodeError:
        return default or {}

def safe_json_dumps(obj: Any) -> str:
    """Safely dump object to JSON string"""
    try:
        return json.dumps(obj)
    except (TypeError, ValueError):
        return "{}"
