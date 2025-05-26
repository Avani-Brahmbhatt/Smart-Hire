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