#!/usr/bin/env python3
"""
AI Hiring Agent Command Line Interface
A comprehensive CLI tool for managing the AI hiring process
"""

import argparse
import sys
import os
from datetime import datetime, timedelta
from typing import List, Dict, Any
import json
from tabulate import tabulate

# Import the main system (assuming it's in the same directory)
try:
    from main_system import AIHiringAgent
    from utils import logger
except ImportError:
    print("Error: Could not import main_system.py. Make sure it's in the same directory.")
    sys.exit(1)


class AIHiringCLI:
    def __init__(self):
        self.agent = AIHiringAgent()
        print("🤖 AI Hiring Agent CLI initialized successfully!")
    
    def display_header(self, title: str):
        """Display a formatted header"""
        print("\n" + "="*60)
        print(f" {title}")
        print("="*60)
    
    def display_table(self, data: List[Dict], headers: List[str] = None):
        """Display data in a formatted table"""
        if not data:
            print("No data to display.")
            return
        
        if headers is None:
            headers = list(data[0].keys()) if data else []
        
        table_data = []
        for item in data:
            row = [str(item.get(header, "N/A")) for header in headers]
            table_data.append(row)
        
        print(tabulate(table_data, headers=headers, tablefmt="grid"))
    
    def get_system_stats(self):
        """Display system statistics"""
        self.display_header("SYSTEM STATISTICS")
        
        stats = self.agent.get_system_stats()
        
        print(f"📊 Total Candidates: {stats['total_candidates']}")
        print(f"💼 Total Jobs: {stats['total_jobs']}")
        print(f"🤝 Total Interviews: {stats['total_interviews']}")
        print(f"📈 Recent Candidates (30 days): {stats['recent_candidates']}")
        print(f"🆕 Recent Jobs (30 days): {stats['recent_jobs']}")
        print(f"🔍 Vectorstore Available: {'✅' if stats['vectorstore_available'] else '❌'}")
    
    def ingest_resumes(self, folder_path: str = None):
        """Ingest resumes from a folder"""
        self.display_header("RESUME INGESTION")
        
        folder = folder_path if folder_path else input("Enter resume folder path (or press Enter for default): ").strip()
        
        if not folder:
            folder = None  # Use default from config
        
        print(f"🔄 Ingesting resumes from: {folder or 'default folder'}")
        
        try:
            candidates_added = self.agent.ingest_resumes(folder)
            print(f"✅ Successfully added {candidates_added} new candidates!")
        except Exception as e:
            print(f"❌ Error ingesting resumes: {str(e)}")
    
    def add_job(self):
        """Add a new job posting"""
        self.display_header("ADD NEW JOB")
        
        print("Enter job details:")
        title = input("Job Title: ").strip()
        description = input("Job Description: ").strip()
        requirements = input("Requirements (optional): ").strip()
        department = input("Department (optional): ").strip()
        location = input("Location (optional): ").strip()
        salary_range = input("Salary Range (optional): ").strip()
        
        if not title or not description:
            print("❌ Title and description are required!")
            return None
        
        try:
            job_id = self.agent.add_job(
                title=title,
                description=description,
                requirements=requirements or None,
                department=department or None,
                location=location or None,
                salary_range=salary_range or None
            )
            
            if job_id:
                print(f"✅ Job added successfully! ID: {job_id}")
                return job_id
            else:
                print("❌ Failed to add job!")
                return None
        except Exception as e:
            print(f"❌ Error adding job: {str(e)}")
            return None
    
    def match_candidates_to_job(self, job_id: str = None, top_k: int = 5):
        """Match candidates to a job"""
        self.display_header("CANDIDATE MATCHING")
        
        if not job_id:
            job_id = input("Enter Job ID: ").strip()
        
        if not job_id:
            print("❌ Job ID is required!")
            return
        
        try:
            print(f"🔍 Finding top {top_k} candidates for job {job_id}...")
            matches = self.agent.match_candidates_to_job(job_id, top_k)
            
            if matches:
                print(f"✅ Found {len(matches)} matching candidates:\n")
                
                # Prepare data for table display
                table_data = []
                for match in matches:
                    skills_str = ", ".join(match['skills'][:3]) + ("..." if len(match['skills']) > 3 else "")
                    table_data.append({
                        "Name": match['name'],
                        "Email": match['email'],
                        "Experience": f"{match['experience_years']} years",
                        "Score": f"{match['similarity_score']:.3f}",
                        "Top Skills": skills_str
                    })
                
                self.display_table(table_data)
                return matches
            else:
                print("❌ No matching candidates found!")
                return []
        except Exception as e:
            print(f"❌ Error matching candidates: {str(e)}")
            return []
    
    def schedule_interview(self):
        """Schedule an interview"""
        self.display_header("SCHEDULE INTERVIEW")
        
        candidate_id = input("Enter Candidate ID: ").strip()
        job_id = input("Enter Job ID: ").strip()
        interviewer_email = input("Enter Interviewer Email: ").strip()
        
        print("\nEnter interview date and time:")
        date_str = input("Date (YYYY-MM-DD): ").strip()
        time_str = input("Time (HH:MM): ").strip()
        
        meeting_link = input("Meeting Link (optional): ").strip()
        
        try:
            # Parse datetime
            datetime_str = f"{date_str} {time_str}"
            interview_time = datetime.strptime(datetime_str, "%Y-%m-%d %H:%M")
            
            success = self.agent.schedule_interview(
                candidate_id=candidate_id,
                job_id=job_id,
                interviewer_email=interviewer_email,
                interview_time=interview_time,
                meeting_link=meeting_link or None
            )
            
            if success:
                print("✅ Interview scheduled successfully!")
                print(f"📅 Date & Time: {interview_time.strftime('%Y-%m-%d %H:%M')}")
            else:
                print("❌ Failed to schedule interview!")
        except ValueError:
            print("❌ Invalid date/time format!")
        except Exception as e:
            print(f"❌ Error scheduling interview: {str(e)}")
    
    def process_video_interview(self):
        """Process a video interview"""
        self.display_header("PROCESS VIDEO INTERVIEW")
        
        candidate_id = input("Enter Candidate ID: ").strip()
        video_path = input("Enter Video File Path: ").strip()
        
        if not os.path.exists(video_path):
            print("❌ Video file does not exist!")
            return
        
        try:
            print("🎥 Processing video interview...")
            transcript = self.agent.process_video_interview(candidate_id, video_path)
            
            if transcript:
                print("✅ Video processed successfully!")
                print(f"\n📝 Transcript Preview:\n{transcript[:500]}...")
            else:
                print("❌ Failed to process video!")
        except Exception as e:
            print(f"❌ Error processing video: {str(e)}")
    
    def send_bulk_rejections(self):
        """Send bulk rejection emails"""
        self.display_header("SEND BULK REJECTIONS")
        
        job_id = input("Enter Job ID: ").strip()
        
        print("Enter candidate IDs to reject (comma-separated):")
        candidate_ids_str = input("Candidate IDs: ").strip()
        
        if not job_id or not candidate_ids_str:
            print("❌ Job ID and candidate IDs are required!")
            return
        
        candidate_ids = [id.strip() for id in candidate_ids_str.split(",")]
        
        try:
            success_count = self.agent.send_bulk_rejection_emails(job_id, candidate_ids)
            print(f"✅ Sent {success_count} rejection emails successfully!")
        except Exception as e:
            print(f"❌ Error sending rejection emails: {str(e)}")
    
    def get_candidate_insights(self):
        """Get AI insights about a candidate"""
        self.display_header("CANDIDATE INSIGHTS")
        
        print("Search by:")
        print("1. Candidate ID")
        print("2. Candidate Name")
        
        choice = input("Choose option (1 or 2): ").strip()
        
        try:
            if choice == "1":
                candidate_id = input("Enter Candidate ID: ").strip()
                insights = self.agent.get_candidate_insights(candidate_id=candidate_id)
            elif choice == "2":
                candidate_name = input("Enter Candidate Name: ").strip()
                insights = self.agent.get_candidate_insights(candidate_name=candidate_name)
            else:
                print("❌ Invalid choice!")
                return
            
            print(f"\n🔍 AI Insights:\n{insights}")
        except Exception as e:
            print(f"❌ Error getting insights: {str(e)}")
    
    def find_candidates_with_skills(self):
        """Find candidates with specific skills"""
        self.display_header("FIND CANDIDATES BY SKILLS")
        
        skills_str = input("Enter skills (comma-separated): ").strip()
        
        if not skills_str:
            print("❌ Skills are required!")
            return
        
        skills = [skill.strip() for skill in skills_str.split(",")]
        
        try:
            result = self.agent.find_candidates_with_skills(skills)
            print(f"\n🔍 Search Results:\n{result}")
        except Exception as e:
            print(f"❌ Error finding candidates: {str(e)}")
    
    def compare_candidates(self):
        """Compare two candidates"""
        self.display_header("COMPARE CANDIDATES")
        
        candidate1_id = input("Enter First Candidate ID: ").strip()
        candidate2_id = input("Enter Second Candidate ID: ").strip()
        
        if not candidate1_id or not candidate2_id:
            print("❌ Both candidate IDs are required!")
            return
        
        try:
            comparison = self.agent.compare_candidates(candidate1_id, candidate2_id)
            print(f"\n⚖️ Comparison Results:\n{comparison}")
        except Exception as e:
            print(f"❌ Error comparing candidates: {str(e)}")
    
    def get_job_analytics(self):
        """Get analytics for a specific job"""
        self.display_header("JOB ANALYTICS")
        
        job_id = input("Enter Job ID: ").strip()
        
        if not job_id:
            print("❌ Job ID is required!")
            return
        
        try:
            analytics = self.agent.get_job_analytics(job_id)
            
            if "error" in analytics:
                print(f"❌ {analytics['error']}")
                return
            
            print(f"📊 Job Analytics for: {analytics['job_title']}")
            print(f"👥 Total Applicants: {analytics['total_applicants']}")
            print(f"📈 Average Score: {analytics['average_score']}")
            
            if analytics.get('status_breakdown'):
                print("\n📋 Status Breakdown:")
                for status, count in analytics['status_breakdown'].items():
                    print(f"  • {status}: {count}")
            
            if analytics.get('top_candidates'):
                print("\n🏆 Top Candidates:")
                self.display_table(analytics['top_candidates'])
            
        except Exception as e:
            print(f"❌ Error getting job analytics: {str(e)}")
    
    def ask_question(self):
        """Ask a question using the RAG system"""
        self.display_header("ASK AI ASSISTANT")
        
        question = input("Enter your question: ").strip()
        
        if not question:
            print("❌ Question is required!")
            return
        
        try:
            print("🤔 Thinking...")
            answer = self.agent.ask_question(question)
            print(f"\n💡 Answer:\n{answer}")
        except Exception as e:
            print(f"❌ Error getting answer: {str(e)}")
    
    def interactive_demo(self):
        """Run an interactive demo with sample data"""
        self.display_header("INTERACTIVE DEMO")
        
        print("🚀 Running interactive demo...")
        
        # 1. Show system stats
        self.get_system_stats()
        
        # 2. Ingest resumes if needed
        print("\n🔄 Checking for resumes to ingest...")
        try:
            candidates_added = self.agent.ingest_resumes()
            if candidates_added > 0:
                print(f"✅ Added {candidates_added} new candidates!")
            else:
                print("ℹ️ No new candidates to add.")
        except Exception as e:
            print(f"⚠️ Resume ingestion skipped: {str(e)}")
        
        # 3. Add a sample job
        print("\n💼 Adding sample job...")
        job_id = self.agent.add_job(
            title="Senior Python Developer",
            description="We are looking for an experienced Python developer to join our team.",
            requirements="5+ years Python experience, Django/Flask, AWS, SQL",
            department="Engineering",
            location="Remote",
            salary_range="$100k-$130k"
        )
        
        if job_id:
            print(f"✅ Sample job added with ID: {job_id}")
            
            # 4. Match candidates
            print(f"\n🔍 Matching candidates to job...")
            matches = self.agent.match_candidates_to_job(job_id, top_k=3)
            
            if matches:
                print(f"✅ Found {len(matches)} matching candidates!")
                table_data = []
                for match in matches:
                    table_data.append({
                        "Name": match['name'],
                        "Score": f"{match['similarity_score']:.3f}",
                        "Experience": f"{match['experience_years']} years"
                    })
                self.display_table(table_data)
                
                # 5. Show job analytics
                print(f"\n📊 Getting job analytics...")
                analytics = self.agent.get_job_analytics(job_id)
                if "error" not in analytics:
                    print(f"📈 Average Score: {analytics['average_score']}")
                    print(f"👥 Total Applicants: {analytics['total_applicants']}")
            else:
                print("ℹ️ No candidates found to match.")
        
        # 6. Test RAG system
        if self.agent.rag_chatbot.rag_chain:
            print("\n🤖 Testing AI Assistant...")
            sample_questions = [
                "What candidates have Python experience?",
                "Who has the most experience?",
                "What are the most common skills?"
            ]
            
            for question in sample_questions:
                try:
                    answer = self.agent.ask_question(question)
                    print(f"\n❓ Q: {question}")
                    print(f"💡 A: {answer[:200]}...")
                except Exception as e:
                    print(f"⚠️ Could not answer: {question}")
        
        print("\n🎉 Demo completed successfully!")


def main():
    parser = argparse.ArgumentParser(description="AI Hiring Agent CLI")
    parser.add_argument('--demo', action='store_true', help='Run interactive demo')
    
    if len(sys.argv) == 1:
        # No arguments provided, show interactive menu
        cli = AIHiringCLI()
        
        while True:
            print("\n" + "="*60)
            print(" 🤖 AI HIRING AGENT - MAIN MENU")
            print("="*60)
            print("1.  📊 System Statistics")
            print("2.  📁 Ingest Resumes")
            print("3.  💼 Add Job")
            print("4.  🔍 Match Candidates to Job")
            print("5.  📅 Schedule Interview")
            print("6.  🎥 Process Video Interview")
            print("7.  ✉️  Send Bulk Rejections")
            print("8.  🔍 Get Candidate Insights")
            print("9.  🎯 Find Candidates by Skills")
            print("10. ⚖️  Compare Candidates")
            print("11. 📊 Job Analytics")
            print("12. 🤔 Ask AI Assistant")
            print("13. 🚀 Run Demo")
            print("0.  🚪 Exit")
            
            try:
                choice = input("\nEnter your choice (0-13): ").strip()
                
                if choice == "0":
                    print("👋 Goodbye!")
                    break
                elif choice == "1":
                    cli.get_system_stats()
                elif choice == "2":
                    cli.ingest_resumes()
                elif choice == "3":
                    cli.add_job()
                elif choice == "4":
                    cli.match_candidates_to_job()
                elif choice == "5":
                    cli.schedule_interview()
                elif choice == "6":
                    cli.process_video_interview()
                elif choice == "7":
                    cli.send_bulk_rejections()
                elif choice == "8":
                    cli.get_candidate_insights()
                elif choice == "9":
                    cli.find_candidates_with_skills()
                elif choice == "10":
                    cli.compare_candidates()
                elif choice == "11":
                    cli.get_job_analytics()
                elif choice == "12":
                    cli.ask_question()
                elif choice == "13":
                    cli.interactive_demo()
                else:
                    print("❌ Invalid choice! Please try again.")
                
                input("\nPress Enter to continue...")
                
            except KeyboardInterrupt:
                print("\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"❌ An error occurred: {str(e)}")
                input("\nPress Enter to continue...")
        
        # Cleanup
        try:
            cli.agent.cleanup()
        except:
            pass
    
    else:
        # Handle command line arguments
        args = parser.parse_args()
        
        if args.demo:
            cli = AIHiringCLI()
            cli.interactive_demo()
            cli.agent.cleanup()


if __name__ == "__main__":
    main()