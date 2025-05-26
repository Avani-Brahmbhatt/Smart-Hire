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

# -----------------------------
# 2. Load Resume & JD Text
# -----------------------------
resume_path = "Avani_Brahmbhatt_Resume.pdf"  # change to your path
jd_path = "salesjd.txt"

resume_text = extract_text(resume_path)
jd_text = extract_text(jd_path)

# -----------------------------
# 3. Enhanced Text Chunking for Better Section Recognition
# -----------------------------
# Use larger chunks with more overlap to preserve context
splitter = RecursiveCharacterTextSplitter(
    chunk_size=800, 
    chunk_overlap=200,
    separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""]
)

# Create resume chunks with section identification
resume_chunks = splitter.create_documents([resume_text])
jd_chunks = splitter.create_documents([jd_text])

# Enhanced metadata for resume chunks - identify sections
import re
for i, doc in enumerate(resume_chunks):
    doc.metadata["source"] = "resume"
    doc.metadata["chunk_id"] = i
    
    # Identify if chunk contains projects or experience
    content_lower = doc.page_content.lower()
    if any(keyword in content_lower for keyword in ['project', 'developed', 'built', 'created', 'implemented', 'designed']):
        doc.metadata["contains_projects"] = True
    if any(keyword in content_lower for keyword in ['experience', 'worked', 'company', 'position', 'role', 'employed', 'intern']):
        doc.metadata["contains_experience"] = True

for doc in jd_chunks:
    doc.metadata["source"] = "job_description"

# Also create a comprehensive resume document for complete context
full_resume_doc = splitter.create_documents([f"COMPLETE RESUME CONTEXT:\n{resume_text}"])
for doc in full_resume_doc:
    doc.metadata["source"] = "resume_full"

all_chunks = resume_chunks + jd_chunks + full_resume_doc

# -----------------------------
# 4. Create Vector DB with AllMiniLM
# -----------------------------
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vectorstore = FAISS.from_documents(all_chunks, embedding_model)

# -----------------------------
# 5. RAG with Enhanced Retrieval for Projects/Experience
# -----------------------------
groq_api_key = os.getenv("GROQ_API_KEY")  # Or replace with a string but don't hardcode secrets

llm = ChatGroq(
    api_key=groq_api_key,
    model_name="llama-3.1-8b-instant",
    temperature=0.1,  # Lower temperature for more consistent extraction
    # Disable thinking process output for DeepSeek
    extra_body={"response_format": {"type": "text"}}
)

# Enhanced retriever to get more relevant chunks
retriever = vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 8}  # Retrieve more chunks to ensure we get projects/experience
)

qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    return_source_documents=True
)

# -----------------------------
# 6. Multi-Step Approach for Better Project/Experience Extraction
# -----------------------------

# First, let's extract projects and experience directly from the full resume text
def extract_projects_and_experience(resume_text):
    """Extract projects and experience sections from resume text"""
    projects = []
    experience = []
    
    lines = resume_text.split('\n')
    current_section = None
    current_content = []
    
    for line in lines:
        line_lower = line.lower().strip()
        
        # Detect section headers
        if any(keyword in line_lower for keyword in ['project', 'projects']):
            if current_section and current_content:
                if current_section == 'projects':
                    projects.extend(current_content)
                elif current_section == 'experience':
                    experience.extend(current_content)
            current_section = 'projects'
            current_content = [line.strip()]
            
        elif any(keyword in line_lower for keyword in ['experience', 'work experience', 'employment', 'career']):
            if current_section and current_content:
                if current_section == 'projects':
                    projects.extend(current_content)
                elif current_section == 'experience':
                    experience.extend(current_content)
            current_section = 'experience'
            current_content = [line.strip()]
            
        elif line.strip() and current_section:
            current_content.append(line.strip())
    
    # Add remaining content
    if current_section and current_content:
        if current_section == 'projects':
            projects.extend(current_content)
        elif current_section == 'experience':
            experience.extend(current_content)
    
    return projects, experience

# Extract projects and experience
extracted_projects, extracted_experience = extract_projects_and_experience(resume_text)

# Enhanced prompt with pre-extracted information and specific instructions
prompt = f"""
Based on the provided resume and job description, conduct a comprehensive eligibility assessment. 
The resume contains the following information that you must include in your analysis:

RESUME PROJECTS SECTION:
{chr(10).join(extracted_projects) if extracted_projects else "No specific projects section found"}

RESUME EXPERIENCE SECTION:
{chr(10).join(extracted_experience) if extracted_experience else "No specific experience section found"}

INSTRUCTIONS:
- Analyze the complete resume content for projects, work experience, skills, and qualifications
- Look for project details, technologies used, achievements, and work history
- Match these against the job requirements
- Provide ONLY the final analysis without any thinking process or reasoning steps

Format your response exactly as follows:

**ELIGIBILITY SCORE:** [X/10]

**ELIGIBILITY STATUS:** [Perfect Match/Strong Match/Good Match/Weak Match/Poor Match]

**MATCHED CRITERIA:**
- [List specific job requirements that the candidate meets based on resume content]
- [Include skills, qualifications, experience levels, technologies, etc.]

**MISSING CRITERIA:**
- [List specific job requirements the candidate doesn't meet]
- [Include any gaps in skills, experience, or qualifications]

**CANDIDATE'S PROJECTS:**
- [Extract and list ALL projects mentioned in the resume with brief descriptions]
- [Include technologies used, scope, and achievements if mentioned]
- [If no projects found, state "No projects explicitly mentioned"]

**WORK EXPERIENCE:**
- [Extract and list ALL work experience/positions from the resume]
- [Include company names, roles, duration, and key responsibilities if mentioned]
- [If no work experience found, state "No work experience explicitly mentioned"]

**OVERALL ASSESSMENT:**
[2-3 sentences summarizing the candidate's fit for the role based on extracted information]

IMPORTANT: You must extract projects and work experience from the resume content. Look carefully through all sections.
"""

# Use invoke instead of run (fixes the main error)
result = qa_chain.invoke({"query": prompt})

# Extract and clean the response to remove any thinking process
response_text = result["result"]

# Filter out DeepSeek thinking process (usually enclosed in <think> tags or similar patterns)
import re
# Remove any thinking process patterns commonly used by DeepSeek
response_text = re.sub(r'<think>.*?</think>', '', response_text, flags=re.DOTALL)
response_text = re.sub(r'<thinking>.*?</thinking>', '', response_text, flags=re.DOTALL)
response_text = re.sub(r'\*\*Thinking:.*?\*\*Answer:', '**Answer:', response_text, flags=re.DOTALL)
response_text = response_text.strip()

# -----------------------------
# 7. Enhanced Output Display
# -----------------------------
print("="*60)
print("           RESUME ELIGIBILITY ASSESSMENT")
print("="*60)
print(response_text)
# print("\n" + "="*60)
# print("                SOURCE ANALYSIS")  
# print("="*60)

# # Group and display source documents with enhanced project/experience detection
# resume_sources = []
# jd_sources = []
# project_sources = []
# experience_sources = []

# for doc in result["source_documents"]:
#     source_type = doc.metadata.get('source', 'unknown')
#     content_preview = doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content
    
#     if source_type == 'resume' or source_type == 'resume_full':
#         resume_sources.append(content_preview)
        
#         # Check if this chunk contains projects or experience
#         if doc.metadata.get('contains_projects'):
#             project_sources.append(content_preview)
#         if doc.metadata.get('contains_experience'):
#             experience_sources.append(content_preview)
            
#     elif source_type == 'job_description':
#         jd_sources.append(content_preview)

# if resume_sources:
#     print("\n📄 RESUME EXCERPTS ANALYZED:")
#     for i, excerpt in enumerate(resume_sources, 1):
#         print(f"  {i}. {excerpt}")

# if project_sources:
#     print("\n🚀 PROJECT-RELATED SECTIONS FOUND:")
#     for i, excerpt in enumerate(project_sources, 1):
#         print(f"  {i}. {excerpt}")

# if experience_sources:
#     print("\n💼 EXPERIENCE-RELATED SECTIONS FOUND:")
#     for i, excerpt in enumerate(experience_sources, 1):
#         print(f"  {i}. {excerpt}")

# if jd_sources:
#     print("\n📋 JOB DESCRIPTION EXCERPTS ANALYZED:")
#     for i, excerpt in enumerate(jd_sources, 1):
#         print(f"  {i}. {excerpt}")

# # Display the extracted sections for debugging
# if extracted_projects:
#     print(f"\n🔍 DIRECTLY EXTRACTED PROJECTS ({len(extracted_projects)} items):")
#     for i, project in enumerate(extracted_projects[:5], 1):  # Show first 5
#         print(f"  {i}. {project}")

# if extracted_experience:
#     print(f"\n🔍 DIRECTLY EXTRACTED EXPERIENCE ({len(extracted_experience)} items):")
#     for i, exp in enumerate(extracted_experience[:5], 1):  # Show first 5
#         print(f"  {i}. {exp}")

# print("\n" + "="*60)