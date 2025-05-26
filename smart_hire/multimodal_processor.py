# multimodal_processor.py
import os
from typing import Optional
import whisper
from config import Config
from utils import logger, ensure_directory_exists

class MultimodalProcessor:
    def __init__(self):
        self.whisper_model = None
        ensure_directory_exists(Config.UPLOAD_FOLDER)
   
    def load_whisper_model(self):
        """Lazy load Whisper model"""
        if self.whisper_model is None:
            try:
                self.whisper_model = whisper.load_model(Config.WHISPER_MODEL)
                logger.info(f"Loaded Whisper model: {Config.WHISPER_MODEL}")
            except Exception as e:
                logger.error(f"Error loading Whisper model: {str(e)}")
   
    def transcribe_audio(self, file_path: str) -> Optional[str]:
        """Transcribe audio/video file"""
        if not os.path.exists(file_path):
            logger.error(f"File not found: {file_path}")
            return None
       
        self.load_whisper_model()
        if not self.whisper_model:
            logger.error("Whisper model not available")
            return None
       
        try:
            result = self.whisper_model.transcribe(file_path)
            transcript = result["text"].strip()
            logger.info(f"Transcribed audio file: {file_path}")
            return transcript
        except Exception as e:
            logger.error(f"Error transcribing file {file_path}: {str(e)}")
            return None
   
    def process_candidate_video(self, video_file_path: str, candidate_id: str) -> Optional[str]:
        """Process candidate video interview"""
        transcript = self.transcribe_audio(video_file_path)
        if transcript:
            # Save transcript
            transcript_file = os.path.join(
                Config.UPLOAD_FOLDER,
                f"{candidate_id}_transcript.txt"
            )
            try:
                with open(transcript_file, 'w', encoding='utf-8') as f:
                    f.write(transcript)
                logger.info(f"Saved transcript: {transcript_file}")
            except Exception as e:
                logger.error(f"Error saving transcript: {str(e)}")
       
        return transcript
