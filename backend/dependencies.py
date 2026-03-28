from sqlalchemy.orm import Session
from fastapi import Depends, HTTPException, status
from backend.database import SessionLocal
from backend import auth, models

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

async def get_current_user(token: str = Depends(auth.oauth2_scheme), db: Session = Depends(get_db)):
    return auth.get_current_user(token, db)
