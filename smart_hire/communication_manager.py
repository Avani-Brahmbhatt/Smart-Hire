# communication_manager.py
import smtplib
from email.message import EmailMessage
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from typing import List, Optional
from datetime import datetime, timedelta
import json
from config import Config
from utils import logger

# For Google Calendar (optional - requires google-api-python-client)
try:
    from google.oauth2 import service_account
    from googleapiclient.discovery import build
    GOOGLE_CALENDAR_AVAILABLE = True
except ImportError:
    GOOGLE_CALENDAR_AVAILABLE = False
    logger.warning("Google Calendar API not available. Install google-api-python-client to enable scheduling.")

class CommunicationManager:
    def __init__(self):
        self.email = Config.GMAIL_EMAIL
        self.password = Config.GMAIL_PASSWORD
        self.service_account_file = Config.GOOGLE_SERVICE_ACCOUNT_FILE
   
    def send_email(self, to_email: str, subject: str, body: str,
                   cc_emails: List[str] = None, is_html: bool = False) -> bool:
        """Send email using Gmail SMTP"""
        try:
            if is_html:
                msg = MIMEMultipart('alternative')
                msg['Subject'] = subject
                msg['From'] = self.email
                msg['To'] = to_email
                if cc_emails:
                    msg['Cc'] = ', '.join(cc_emails)
               
                html_part = MIMEText(body, 'html')
                msg.attach(html_part)
            else:
                msg = EmailMessage()
                msg.set_content(body)
                msg['Subject'] = subject
                msg['From'] = self.email
                msg['To'] = to_email
                if cc_emails:
                    msg['Cc'] = ', '.join(cc_emails)
           
            with smtplib.SMTP_SSL('smtp.gmail.com', 465) as smtp:
                smtp.login(self.email, self.password)
                recipients = [to_email] + (cc_emails or [])
                smtp.send_message(msg, to_addrs=recipients)
           
            logger.info(f"Email sent successfully to {to_email}")
            return True
           
        except Exception as e:
            logger.error(f"Error sending email: {str(e)}")
            return False
   
    def send_interview_invitation(self, candidate_email: str, candidate_name: str,
                                job_title: str, interview_time: datetime,
                                meeting_link: str = None) -> bool:
        """Send interview invitation email"""
        subject = f"Interview Invitation - {job_title} Position"
       
        body = f"""
        Dear {candidate_name},

        Thank you for your interest in the {job_title} position. We are pleased to invite you for an interview.

        Interview Details:
        - Date & Time: {interview_time.strftime("%A, %B %d, %Y at %I:%M %p")}
        - Duration: 60 minutes
        """
       
        if meeting_link:
            body += f"- Meeting Link: {meeting_link}\n"
       
        body += """
        Please confirm your availability by replying to this email.

        Best regards,
        HR Team
        """
       
        return self.send_email(candidate_email, subject, body)
   
    def send_rejection_email(self, candidate_email: str, candidate_name: str,
                           job_title: str) -> bool:
        """Send rejection email"""
        subject = f"Application Status - {job_title} Position"
       
        body = f"""
        Dear {candidate_name},

        Thank you for your interest in the {job_title} position and for taking the time to apply.

        After careful consideration of your application, we have decided to move forward with other candidates who more closely match our current requirements.

        We appreciate your interest in our company and encourage you to apply for future positions that match your skills and experience.

        Best regards,
        HR Team
        """
       
        return self.send_email(candidate_email, subject, body)
   
    def schedule_google_calendar_event(self, summary: str, start_time: datetime,
                                     end_time: datetime, attendee_emails: List[str],
                                     description: str = "") -> Optional[str]:
        """Schedule Google Calendar event"""
        if not GOOGLE_CALENDAR_AVAILABLE:
            logger.error("Google Calendar API not available")
            return None
       
        try:
            scopes = ['https://www.googleapis.com/auth/calendar']
            creds = service_account.Credentials.from_service_account_file(
                self.service_account_file, scopes=scopes
            )
            service = build('calendar', 'v3', credentials=creds)
           
            event = {
                'summary': summary,
                'description': description,
                'start': {
                    'dateTime': start_time.isoformat(),
                    'timeZone': 'UTC',
                },
                'end': {
                    'dateTime': end_time.isoformat(),
                    'timeZone': 'UTC',
                },
                'attendees': [{'email': email} for email in attendee_emails],
                'reminders': {
                    'useDefault': False,
                    'overrides': [
                        {'method': 'email', 'minutes': 24 * 60},
                        {'method': 'popup', 'minutes': 10},
                    ],
                },
            }
           
            event = service.events().insert(calendarId='primary', body=event).execute()
            logger.info(f"Calendar event created: {event.get('htmlLink')}")
            return event.get('htmlLink')
           
        except Exception as e:
            logger.error(f"Error creating calendar event: {str(e)}")
            return None
