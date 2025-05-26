# database.py
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from models import Base
from config import Config

class DatabaseManager:
    def __init__(self):
        self.engine = create_engine(Config.DATABASE_URL)
        Base.metadata.create_all(self.engine)
        Session = sessionmaker(bind=self.engine)
        self.session = Session()
   
    def get_session(self):
        return self.session
   
    def close(self):
        self.session.close()

