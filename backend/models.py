from sqlalchemy import Column, Integer, String, DateTime
from datetime import datetime
from backend.database import Base

class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    username = Column(String, unique=True, index=True)
    hashed_password = Column(String)

class ProjectLog(Base):
    __tablename__ = "project_logs"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer)
    username = Column(String)
    filename = Column(String)
    timestamp = Column(DateTime, default=datetime.utcnow)
    action = Column(String)
