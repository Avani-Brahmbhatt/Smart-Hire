# AI Hiring Agent - Complete Setup Guide

## 📋 Prerequisites

- Python 3.8 or higher
- Git (optional, for cloning)
- Gmail account (for email functionality)
- Groq API account (for AI/LLM features)
- Google Cloud account (optional, for calendar integration)

## 🛠 Step 1: Environment Setup

### 1.1 Create Project Directory
```bash
mkdir ai-hiring-agent
cd ai-hiring-agent
```

### 1.2 Create Virtual Environment
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

### 1.3 Create Project Structure
```bash
# Create directories
mkdir resumes uploads faiss_index

# Create files
touch config.py
touch utils.py
touch models.py
touch database.py
touch resume_processor.py
touch embedding_manager.py
touch job_matcher.py
touch rag_chatbot.py
touch communication_manager.py
touch multimodal_processor.py
touch main_system.py
touch requirements.txt
touch .env
touch run_demo.py
```

## 📦 Step 2: Install Dependencies

### 2.1 Create requirements.txt
```txt
# Core dependencies
langchain==0.1.0
langchain-community==0.0.13
langchain-groq==0.0.3
faiss-cpu==1.7.4
sqlalchemy==2.0.25
sentence-transformers==2.2.2

# Document processing
pypdf==3.17.4
python-docx==1.1.0
docx2txt==0.8

# AI/ML
openai-whisper==20231117
scikit-learn==1.3.2
numpy==1.24.3

# Google services (optional)
google-api-python-client==2.116.0
google-auth-httplib2==0.2.0
google-auth-oauthlib==1.2.0

# Utilities
python-dotenv==1.0.0
requests==2.31.0
```

### 2.2 Install Packages
```bash
pip install -r requirements.txt
```

## 🔧 Step 3: Configuration Setup

### 3.1 Create .env file
```env
# API Keys
GROQ_API_KEY=your_groq_api_key_here
GMAIL_EMAIL=your_email@gmail.com
GMAIL_PASSWORD=your_app_password_here

# Directories
RESUME_FOLDER=resumes
FAISS_INDEX_DIR=faiss_index
UPLOAD_FOLDER=uploads

# Database
DATABASE_URL=sqlite:///hiring_agent.db

# Optional: Google Calendar
GOOGLE_SERVICE_ACCOUNT_FILE=service_account.json
```

### 3.2 Update config.py
```python
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class Config:
    """Configuration management for the AI Hiring Agent"""
   
    # Directories
    RESUME_FOLDER = os.getenv("RESUME_FOLDER", "resumes")
    FAISS_INDEX_DIR = os.getenv("FAISS_INDEX_DIR", "faiss_index")
    UPLOAD_FOLDER = os.getenv("UPLOAD_FOLDER", "uploads")
    DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///hiring_agent.db")
   
    # API Keys
    GROQ_API_KEY = os.getenv("GROQ_API_KEY")
    GMAIL_PASSWORD = os.getenv("GMAIL_PASSWORD")
    GMAIL_EMAIL = os.getenv("GMAIL_EMAIL")
    GOOGLE_SERVICE_ACCOUNT_FILE = os.getenv("GOOGLE_SERVICE_ACCOUNT_FILE", "service_account.json")
   
    # Model settings
    EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
    LLM_MODEL = "llama3-8b-8192"
    WHISPER_MODEL = "base"
   
    # Processing settings
    CHUNK_SIZE = 500
    CHUNK_OVERLAP = 50
    TOP_K_CANDIDATES = 5
    SIMILARITY_THRESHOLD = 0.3
```

## 🔑 Step 4: API Keys Setup

### 4.1 Get Groq API Key
1. Go to [Groq Console](https://console.groq.com)
2. Sign up/Login
3. Navigate to API Keys section
4. Create new API key
5. Copy and add to `.env` file

### 4.2 Setup Gmail App Password
1. Enable 2-Factor Authentication on your Gmail
2. Go to Google Account settings
3. Security → App passwords
4. Generate app password for "Mail"
5. Use this password in `.env` file (not your regular Gmail password)

### 4.3 Google Calendar Setup (Optional)
1. Go to [Google Cloud Console](https://console.cloud.google.com)
2. Create new project or select existing
3. Enable Google Calendar API
4. Create Service Account
5. Download JSON key file as `service_account.json`
6. Place in project root directory

## 📁 Step 5: Add Sample Data

### 5.1 Add Sample Resumes
Create some sample resume files in the `resumes/` folder:
- `john_doe.pdf`
- `jane_smith.docx`
- `mike_johnson.pdf`

Or use your existing resume files.

## 🚀 Step 6: Create Run Script

### 6.1 Create run_demo.py
```python
#!/usr/bin/env python3
"""
AI Hiring Agent Demo Script
"""
import sys
import os
from datetime import datetime, timedelta

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from main_system import AIHiringAgent

def main():
    print("🤖 Initializing AI Hiring Agent...")
    agent = AIHiringAgent()
    
    try:
        print("\n📊 System Statistics:")
        stats = agent.get_system_stats()
        for key, value in stats.items():
            print(f"  {key}: {value}")
        
        print("\n📄 Ingesting resumes...")
        candidates_added = agent.ingest_resumes()
        print(f"✅ Added {candidates_added} new candidates")
        
        print("\n💼 Creating sample job...")
        job_id = agent.add_job(
            title="Senior Python Developer",
            description="We are looking for an experienced Python developer with expertise in web frameworks and cloud technologies.",
            requirements="5+ years Python experience, Django/Flask, AWS, SQL, REST APIs",
            department="Engineering",
            location="Remote",
            salary_range="$100,000 - $130,000"
        )
        
        if job_id:
            print(f"✅ Created job: {job_id}")
            
            print("\n🎯 Matching candidates to job...")
            matches = agent.match_candidates_to_job(job_id, top_k=5)
            
            if matches:
                print(f"Found {len(matches)} matching candidates:")
                for i, match in enumerate(matches, 1):
                    print(f"  {i}. {match['name']} - Score: {match['similarity_score']:.3f}")
                    print(f"     Email: {match['email']}")
                    print(f"     Skills: {', '.join(match['skills'][:5])}")
                    print(f"     Experience: {match['experience_years']} years\n")
            
            print("📈 Job Analytics:")
            analytics = agent.get_job_analytics(job_id)
            for key, value in analytics.items():
                if key != 'top_candidates':
                    print(f"  {key}: {value}")
        
        print("\n🤔 Testing AI Q&A System...")
        questions = [
            "What candidates have Python experience?",
            "Who has the most experience?",
            "What are the most common skills among candidates?"
        ]
        
        for question in questions:
            print(f"\nQ: {question}")
            answer = agent.ask_question(question)
            print(f"A: {answer[:200]}{'...' if len(answer) > 200 else ''}")
        
        print("\n✅ Demo completed successfully!")
        
        # Interactive mode
        print("\n" + "="*50)
        print("🎯 Interactive Mode - Ask questions about candidates!")
        print("Type 'exit' to quit")
        print("="*50)
        
        while True:
            question = input("\nYour question: ").strip()
            if question.lower() in ['exit', 'quit', 'q']:
                break
            if question:
                answer = agent.ask_question(question)
                print(f"\n💡 Answer: {answer}")
        
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()
    
    finally:
        agent.cleanup()
        print("\n👋 Goodbye!")

if __name__ == "__main__":
    main()
```

## 🏃 Step 7: Running the Project

### 7.1 First Run
```bash
# Make sure virtual environment is activated
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate     # Windows

# Run the demo
python run_demo.py
```

### 7.2 Expected Output
```
🤖 Initializing AI Hiring Agent...
📊 System Statistics:
  total_candidates: 0
  total_jobs: 0
  vectorstore_available: False

📄 Ingesting resumes...
✅ Added 3 new candidates

💼 Creating sample job...
✅ Created job: uuid-string

🎯 Matching candidates to job...
Found 3 matching candidates:
  1. John Doe - Score: 0.751
     Email: john.doe@email.com
     Skills: Python, Django, AWS
     Experience: 5.0 years
...
```

## 🔧 Step 8: Troubleshooting

### Common Issues:

#### 8.1 Import Errors
```bash
# If you get import errors, install missing packages:
pip install langchain-community
pip install sentence-transformers
pip install faiss-cpu
```

#### 8.2 Groq API Issues
- Verify your API key is correct
- Check if you have credits/quota
- Test with a simple request first

#### 8.3 Gmail Issues
- Use App Password, not regular password
- Enable 2FA first
- Check Gmail security settings

#### 8.4 File Permission Issues
```bash
# Make sure directories are writable
chmod 755 resumes uploads faiss_index
```

#### 8.5 Model Download Issues
```bash
# If sentence-transformers model doesn't download:
python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')"
```

## 🎯 Step 9: Usage Examples

### 9.1 Basic Operations
```python
from main_system import AIHiringAgent

agent = AIHiringAgent()

# Add resumes
agent.ingest_resumes()

# Create job
job_id = agent.add_job("Data Scientist", "Looking for ML expert...")

# Match candidates
matches = agent.match_candidates_to_job(job_id)

# Ask questions
answer = agent.ask_question("Who has Python skills?")
```

### 9.2 Advanced Features
```python
# Schedule interview
from datetime import datetime, timedelta
interview_time = datetime.now() + timedelta(days=3)
agent.schedule_interview(candidate_id, job_id, "hr@company.com", interview_time)

# Process video interview
transcript = agent.process_video_interview(candidate_id, "interview.mp4")

# Bulk rejection emails
agent.send_bulk_rejection_emails(job_id, [candidate_id1, candidate_id2])
```

## 📚 Step 10: Next Steps

1. **Customize for your needs**: Modify email templates, add more skill extraction patterns
2. **Add web interface**: Create Flask/FastAPI frontend
3. **Integrate with ATS**: Connect to existing recruitment systems
4. **Add more AI features**: Interview question generation, candidate ranking
5. **Scale up**: Use PostgreSQL, Redis for production

## 💡 Tips for Success

- Start with a few test resumes first
- Test email functionality with your own email
- Monitor Groq API usage to avoid quota limits
- Keep backups of your vectorstore and database
- Use descriptive job descriptions for better matching

## 🆘 Getting Help

If you encounter issues:
1. Check the logs in terminal output
2. Verify all API keys are set correctly
3. Ensure all required files are in place  
4. Test individual components separately
5. Check Python version compatibility

---

**Ready to revolutionize your hiring process? Let's get started! 🚀**