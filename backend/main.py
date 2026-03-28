from fastapi import FastAPI, Depends, HTTPException, status, File, UploadFile, Form
from pydantic import BaseModel
from fastapi.security import OAuth2PasswordRequestForm
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session
import os
import io
from datetime import timedelta
import numpy as np
import pandas as pd

from backend import models, auth, database, eda_engine
from backend.dependencies import get_db, get_current_user
from backend.groq_integration import CHAT_MODEL

# Create tables if not exist
models.Base.metadata.create_all(bind=database.engine)
app = FastAPI(title="AI Powered EDA Wizard API")

class ChatRequest(BaseModel):
    question: str
    eda_summary: str

# Setup CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Authentication endpoints
@app.post("/auth/register")
def register(user: OAuth2PasswordRequestForm = Depends(), db: Session = Depends(database.get_db)):
    try:
        db_user = db.query(models.User).filter(models.User.username == user.username).first()
        if db_user:
            raise HTTPException(status_code=400, detail="Username already registered")
        hashed_password = auth.get_password_hash(user.password)
        new_user = models.User(username=user.username, hashed_password=hashed_password)
        db.add(new_user)
        db.commit()
        db.refresh(new_user)
        return {"msg": "User created successfully"}
    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        print(f"REGISTRATION ERROR: {error_details}")
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {str(e)}")

@app.post("/auth/login")
def login(user: OAuth2PasswordRequestForm = Depends(), db: Session = Depends(database.get_db)):
    db_user = db.query(models.User).filter(models.User.username == user.username).first()
    if not db_user or not auth.verify_password(user.password, db_user.hashed_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    access_token_expires = timedelta(minutes=auth.ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = auth.create_access_token(
        data={"sub": db_user.username}, expires_delta=access_token_expires
    )
    return {"access_token": access_token, "token_type": "bearer"}

# Protected EDA Dashboard Endpoint
@app.post("/eda/upload")
async def upload_dataset(
    file: UploadFile = File(...),
    drop_high_null: bool = Form(False),
    current_user: models.User = Depends(auth.get_current_user),
    db: Session = Depends(database.get_db)
):
    if not (file.filename.endswith(".csv") or file.filename.endswith(".txt")):
        raise HTTPException(status_code=400, detail="Currently only CSV and TXT files are supported.")
    
    # Read the file
    content = await file.read()
    try:
        # Load string stream into pandas dataframe (CSV Only)
        if file.filename.endswith(".csv") or file.filename.endswith(".txt"):
            try:
                # Optimized CSV read with high-performance C engine
                df = pd.read_csv(io.BytesIO(content), engine='c', low_memory=False)
            except:
                df = pd.read_csv(io.BytesIO(content), sep='\t')
        else:
            return JSONResponse(status_code=400, content={"detail": "Only CSV and TXT files are supported."})
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error reading dataset: {str(e)}")

    # Perform Advanced EDA matching Hackfest Part 1 output
    try:
        report, cleaned_df = eda_engine.analyze_and_clean_data(df, drop_high_null=drop_high_null)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Data Transformation Failed: {str(e)}")
    
    # Perform Categorical Univariate Analysis (Part 2)
    try:
        categorical_cols = report["column_types"]["Categorical"]
        cat_analysis = eda_engine.run_categorical_univariate(cleaned_df, categorical_cols)
    except Exception as e:
        cat_analysis = {"error": str(e), "ai_report": "Error", "bar_charts": {}, "pie_charts": {}}
    
    # Perform Numerical Univariate Analysis (Part 3)
    try:
        numerical_cols = report["column_types"]["Numerical"]
        num_analysis = eda_engine.run_numerical_univariate(cleaned_df, numerical_cols)
    except Exception as e:
        num_analysis = {"error": str(e), "describe_table": [], "treatment_logs": []}
    
    # Perform Bivariate Analysis (Part 4)
    try:
        target_col = report["target_column"]
        bivariate_analysis = eda_engine.run_bivariate_analysis(cleaned_df, numerical_cols, categorical_cols, target_col)
    except Exception as e:
        bivariate_analysis = {"error": str(e), "heatmap": "", "ai_report": "Error"}
    
    # Perform Multivariate Analysis (Part 5)
    try:
        multivariate_analysis = eda_engine.run_multivariate_analysis(cleaned_df, numerical_cols, target_col)
    except Exception as e:
        multivariate_analysis = {"error": str(e), "pairplot": "", "ai_report": "Error"}
    
    # Perform Feature Engineering and Selection (Part 6)
    try:
        nominal_cols = report["column_types"]["Nominal"]
        ordinal_cols = report["column_types"]["Ordinal"]
        feature_engineering = eda_engine.run_feature_engineering_and_selection(cleaned_df, nominal_cols, ordinal_cols, target_col)
    except Exception as e:
        feature_engineering = {
            "error": str(e), 
            "feature_importance": [], 
            "selected_columns": [], 
            "encoding_report": [], 
            "feature_chart": "", 
            "final_dataset_b64": ""
        }
    
    # Generate EDA Summary string for AI Assistant (Part 7)
    eda_summary = f"""
Dataset Columns : {list(cleaned_df.columns)}
Target Column : {target_col}
Numerical Columns : {numerical_cols}
Categorical Columns : {categorical_cols}
Important Features : {feature_engineering.get("selected_columns", [])}
"""

    # Part 7: ML Recommendation
    ml_recommendation = eda_engine.run_ml_recommendation(cleaned_df, target_col)

    # Log the action (Part 9: Project Logging)
    try:
        from backend.models import ProjectLog
        new_log = ProjectLog(
            user_id=current_user.id,
            username=current_user.username,
            filename=file.filename,
            action="Started Full EDA Analysis"
        )
        db.add(new_log)
        db.commit()
    except Exception as log_error:
        print(f"Logging error: {log_error}")

    return {
        "filename": file.filename,
        "eda_report": report,
        "categorical_analysis": cat_analysis,
        "numerical_analysis": num_analysis,
        "bivariate_analysis": bivariate_analysis,
        "multivariate_analysis": multivariate_analysis,
        "feature_engineering": feature_engineering,
        "ml_recommendation": ml_recommendation,
        "eda_summary": eda_summary
    }

@app.get("/logs")
def get_user_logs(current_user: models.User = Depends(auth.get_current_user), db: Session = Depends(database.get_db)):
    from backend.models import ProjectLog
    logs = db.query(ProjectLog).filter(ProjectLog.user_id == current_user.id).order_by(ProjectLog.timestamp.desc()).all()
    return logs

@app.post("/eda/chat")
def eda_chat(request: ChatRequest, current_user = Depends(auth.get_current_user)):
    import os
    from groq import Groq
    from fastapi import HTTPException
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key or api_key == "your_groq_api_key_here":
        raise HTTPException(status_code=500, detail="Groq API key not configured.")
        
    client = Groq(api_key=api_key)
    
    prompt = f"""
You are a data science assistant.

Explain answers in SIMPLE ENGLISH.

EDA Summary:
{request.eda_summary}

User Question:
{request.question}
"""
    try:
        chat = client.chat.completions.create(
            model=CHAT_MODEL,
            messages=[{"role":"user", "content": prompt}],
            temperature=0.7
        )
        answer = chat.choices[0].message.content
        return {"answer": answer}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
def health_check():
    return {"status": "AI EDA Backend API is running perfectly!", "version": "HACKFEST_FINAL_V4_MODELS_FIXED"}
