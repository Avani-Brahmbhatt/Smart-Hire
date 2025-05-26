import os
import pdfplumber
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain.schema import Document
from typing import List
import re

# -----------------------------
# 1. Utility: Load File Text
# -----------------------------

def extract_text(file_path: str) -> str:
    """
    Extract text from various file formats using LangChain document loaders.
    
    Args:
        file_path (str): Path to the file to extract text from
        
    Returns:
        str: Extracted text content
        
    Raises:
        ValueError: If file format is not supported
        FileNotFoundError: If file doesn't exist
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    
    file_extension = file_path.lower().split('.')[-1]
    
    try:
        if file_extension == "pdf":
            loader = PyPDFLoader(file_path)
        elif file_extension == "docx":
            loader = Docx2txtLoader(file_path)
        elif file_extension == "txt":
            loader = TextLoader(file_path, encoding="utf-8")
        else:
            raise ValueError(f"Unsupported file format: .{file_extension}")
        
        # Load documents
        documents: List[Document] = loader.load()
        
        # Extract and combine text from all pages/sections
        text_content = "\n".join([doc.page_content for doc in documents])
        
        return text_content
        
    except Exception as e:
        raise Exception(f"Error processing file {file_path}: {str(e)}")

def extract_job_requirements(jd_text: str) -> dict:
    """Extract key components from job description"""
    jd_lower = jd_text.lower()
    
    # Extract job title
    job_title = "Not specified"
    title_patterns = [
        r'job title:?\s*([^\n]+)',
        r'position:?\s*([^\n]+)',
        r'role:?\s*([^\n]+)'
    ]
    for pattern in title_patterns:
        match = re.search(pattern, jd_lower)
        if match:
            job_title = match.group(1).strip()
            break
    
    # Extract experience requirements
    experience_patterns = [
        r'(\d+)[\-\+]*\s*(?:to\s*)?(\d+)?\s*years?\s*(?:of\s*)?experience',
        r'minimum\s*(\d+)\s*years?',
        r'at least\s*(\d+)\s*years?'
    ]
    
    experience_req = "Not specified"
    for pattern in experience_patterns:
        match = re.search(pattern, jd_lower)
        if match:
            if match.group(2):
                experience_req = f"{match.group(1)}-{match.group(2)} years"
            else:
                experience_req = f"{match.group(1)}+ years"
            break
    
    return {
        "job_title": job_title,
        "experience_requirement": experience_req,
        "full_text": jd_text
    }

def extract_resume_summary(resume_text: str) -> dict:
    """Extract key components from resume"""
    resume_lower = resume_text.lower()
    
    # Try to identify the candidate's field/domain
    tech_keywords = ['python', 'java', 'sql', 'programming', 'software', 'data', 'ai', 'ml', 'algorithm']
    sales_keywords = ['sales', 'revenue', 'client', 'customer', 'business development', 'crm']
    marketing_keywords = ['marketing', 'campaign', 'social media', 'branding', 'advertising']
    finance_keywords = ['finance', 'accounting', 'investment', 'banking', 'audit', 'financial']
    
    primary_field = "General"
    if any(keyword in resume_lower for keyword in tech_keywords):
        primary_field = "Technology/Data Science"
    elif any(keyword in resume_lower for keyword in sales_keywords):
        primary_field = "Sales/Business Development"
    elif any(keyword in resume_lower for keyword in marketing_keywords):
        primary_field = "Marketing"
    elif any(keyword in resume_lower for keyword in finance_keywords):
        primary_field = "Finance"
    
    return {
        "primary_field": primary_field,
        "full_text": resume_text
    }

# -----------------------------
# 2. Load Resume & JD Text
# -----------------------------
resume_path = "/home/petpooja-724/Python_Problems/RAG/langchain/smart_hire/resumes/Avani_Brahmbhatt_Resume.pdf"  # change to your path
jd_path = "/home/petpooja-724/Python_Problems/RAG/langchain/smart_hire/job_desc/AIML_GenAI_Engineer_TCS.txt"

resume_text = extract_text(resume_path)
jd_text = extract_text(jd_path)

# Extract summaries
jd_info = extract_job_requirements(jd_text)
resume_info = extract_resume_summary(resume_text)


# -----------------------------
# 3. Text Chunking and Vector Store Setup
# -----------------------------
splitter = RecursiveCharacterTextSplitter(
    chunk_size=700, 
    chunk_overlap=150,
    separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""]
)

resume_chunks = splitter.create_documents([resume_text])
jd_chunks = splitter.create_documents([jd_text])

for i, doc in enumerate(resume_chunks):
    doc.metadata["source"] = "resume"
    doc.metadata["chunk_id"] = i

for doc in jd_chunks:
    doc.metadata["source"] = "job_description"

all_chunks = resume_chunks + jd_chunks

# Create vector store
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vectorstore = FAISS.from_documents(all_chunks, embedding_model)

# -----------------------------
# 4. Setup LLM and RAG Chain
# -----------------------------
groq_api_key = os.getenv("GROQ_API_KEY")  # Set your API key

llm = ChatGroq(
    api_key=groq_api_key,
    model_name="llama-3.1-8b-instant",
    temperature=0.2
)

retriever = vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 8}
)

qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    return_source_documents=True
)

# -----------------------------
# 5. Universal Job-Resume Matching Prompt
# -----------------------------

universal_matching_prompt = f"""
You are an expert HR analyst tasked with evaluating how well a candidate's resume matches a specific job description.

CRITICAL INSTRUCTIONS:
1. Analyze ONLY against the specific job requirements provided
2. Do NOT make assumptions about what the job should require
3. Base your evaluation strictly on what is mentioned in the JD vs what is in the resume
4. Consider field transitions and transferable skills appropriately

JOB DESCRIPTION:
{jd_text}

CANDIDATE'S RESUME:
{resume_text}

ANALYSIS REQUIREMENTS:
Provide a detailed comparison focusing on exact job requirements vs candidate qualifications.

**JOB TITLE & FIELD:** {jd_info['job_title']}

**ELIGIBILITY SCORE:** [X/10] 
(10 = Perfect match for all requirements, 0 = No relevant qualifications)

**ELIGIBILITY STATUS:** [Perfect Match/Strong Match/Good Match/Partial Match/Poor Match/No Match]

**REQUIREMENT ANALYSIS:**

*Essential Requirements Met:*
- [List each essential requirement from JD and whether candidate meets it - be specific]
- [Include education, experience level, technical skills, soft skills as mentioned in JD]
- [Use format: "Requirement: [JD requirement] | Candidate: [What they have] | Match: Yes/No/Partial"]

*Essential Requirements NOT Met:*
- [List requirements from JD that candidate clearly lacks]
- [Be specific about gaps]

*Preferred Requirements Met:*
- [List preferred/nice-to-have requirements candidate meets]

*Preferred Requirements NOT Met:*
- [List preferred requirements candidate lacks]

**EXPERIENCE MATCH:**
- Required: {jd_info['experience_requirement']}
- Candidate Has: [Extract actual experience from resume]
- Match Level: [Exceeds/Meets/Below/Significantly Below]

**SKILLS ALIGNMENT:**
*Technical Skills:*
- Required: [List from JD]
- Candidate Has: [List from resume]
- Gap: [What's missing]

*Soft Skills:*
- Required: [List from JD] 
- Candidate Demonstrates: [Evidence from resume]

**EDUCATION & QUALIFICATIONS:**
- Required: [From JD]
- Candidate Has: [From resume]
- Match: [Yes/No/Alternative qualification]

**FIELD TRANSITION ANALYSIS:**
- Job Field: [Identify from JD]
- Candidate's Background: {resume_info['primary_field']}
- Transition Difficulty: [Same field/Related field/Major transition/Complete career change]
- Transferable Skills: [List skills that apply across fields]

**RED FLAGS:**
- [Any major mismatches, overqualification, underqualification, or concerning gaps]

**OVERALL ASSESSMENT:**
[2-3 sentences on overall fit, considering both direct matches and potential for success in role]

**RECOMMENDATION:**
[Strong Yes/Yes/Maybe/Probably Not/No] - [Brief justification]

Be honest and objective. If it's a poor match, say so clearly with reasons.
"""

# -----------------------------
# 6. Run Analysis
# -----------------------------
print("\nRunning detailed analysis...")
result = qa_chain.invoke({"query": universal_matching_prompt})

# Clean response
response_text = result["result"]
response_text = re.sub(r'<think>.*?</think>', '', response_text, flags=re.DOTALL)
response_text = re.sub(r'<thinking>.*?</thinking>', '', response_text, flags=re.DOTALL)
response_text = response_text.strip()

# -----------------------------
# 7. Display Results
# -----------------------------

print("="*80)
print("                    RESUME-JD MATCHING ANALYSIS")
print("="*80)
print(f"Job Position: {jd_info['job_title']}")  
print(f"Experience Required: {jd_info['experience_requirement']}")
print(f"Candidate's Background: {resume_info['primary_field']}")
print("="*80)
print("                    DETAILED MATCHING ANALYSIS")
print("="*80)
print(response_text)

# -----------------------------
# 8. Quick Compatibility Check
# -----------------------------
print("\n" + "="*80)
print("                    COMPATIBILITY SUMMARY")
print("="*80)

# Simple keyword overlap analysis
jd_words = set(re.findall(r'\b[a-zA-Z]{3,}\b', jd_text.lower()))
resume_words = set(re.findall(r'\b[a-zA-Z]{3,}\b', resume_text.lower()))

# Remove common words
common_words = {'the', 'and', 'for', 'are', 'but', 'not', 'you', 'all', 'can', 'had', 'her', 'was', 'one', 'our', 'out', 'day', 'get', 'has', 'him', 'his', 'how', 'man', 'new', 'now', 'old', 'see', 'two', 'way', 'who', 'boy', 'did', 'its', 'let', 'put', 'say', 'she', 'too', 'use'}
jd_words = jd_words - common_words
resume_words = resume_words - common_words

overlap = jd_words.intersection(resume_words)
overlap_percentage = (len(overlap) / len(jd_words)) * 100 if jd_words else 0

print(f"Keyword Overlap: {overlap_percentage:.1f}%")
print(f"Common Keywords: {', '.join(list(overlap)[:10])}{'...' if len(overlap) > 10 else ''}")
print(f"Field Alignment: {resume_info['primary_field']} → {jd_info['job_title']}")

if overlap_percentage < 20:
    print("⚠️  LOW OVERLAP: Significant skill/domain gap detected")
elif overlap_percentage < 40:
    print("🟡 MODERATE OVERLAP: Some transferable skills present") 
else:
    print("✅ GOOD OVERLAP: Strong keyword alignment")

print("="*80)