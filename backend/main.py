from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from pymongo import MongoClient
from dotenv import load_dotenv
from groq import Groq
import PyPDF2
import os
import io
import json
from datetime import datetime

load_dotenv()

app = FastAPI(title="CVSync")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MONGO_URI = os.getenv("MONGO_URI")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

mongo_client = MongoClient(MONGO_URI)
db = mongo_client["ai_resume_analyzer"]
collection = db["analysis_results"]

groq_client = Groq(api_key=GROQ_API_KEY)


def extract_text_from_pdf(file_bytes):
    pdf_reader = PyPDF2.PdfReader(io.BytesIO(file_bytes))
    text = ""

    for page in pdf_reader.pages:
        text += page.extract_text() or ""

    return text


def analyze_resume_with_groq(resume_text, job_description):
    prompt = f"""
You are a strict ATS resume analyzer.

Analyze the resume against the job description.

Return ONLY valid JSON in this exact format:
{{
  "ats_score": 0,
  "matched_skills": [],
  "missing_skills": [],
  "suggestions": []
}}

Scoring rules:
- Do not give 100 just because one keyword matches.
- Score based on skill match, project relevance, work experience, tools, role fit, and clarity.
- For short job descriptions like "Python developer", infer common expectations:
  Python, backend development, APIs, databases, Git, debugging, problem solving.
- ATS score must be realistic between 0 and 100.
- matched_skills should include only clearly supported skills from the resume.
- missing_skills should include important skills expected for the role but not clearly present.
- suggestions should be practical, simple, and specific.

Resume:
{resume_text[:4000]}

Job Description:
{job_description[:2000]}
"""

    response = groq_client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[
            {
                "role": "system",
                "content": "You are a strict ATS resume evaluator. Return only valid JSON."
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        temperature=0.2,
        response_format={"type": "json_object"}
    )

    text = response.choices[0].message.content
    result = json.loads(text)

    return {
        "source": "Groq AI",
        "ats_score": result.get("ats_score", 0),
        "matched_skills": result.get("matched_skills", []),
        "missing_skills": result.get("missing_skills", []),
        "suggestions": result.get("suggestions", [])
    }


@app.get("/")
def home():
    return {"message": "CVSync backend is running with Groq AI"}


@app.post("/analyze")
async def analyze_resume_api(
    resume: UploadFile = File(...),
    job_description: str = Form(...)
):
    file_bytes = await resume.read()
    resume_text = extract_text_from_pdf(file_bytes)

    result = analyze_resume_with_groq(resume_text, job_description)

    data = {
        "filename": resume.filename,
        "job_description": job_description,
        "resume_text": resume_text[:1000],
        "source": result["source"],
        "ats_score": result["ats_score"],
        "matched_skills": result["matched_skills"],
        "missing_skills": result["missing_skills"],
        "suggestions": result["suggestions"],
        "created_at": datetime.now()
    }

    collection.insert_one(data)

    data["_id"] = str(data["_id"])
    data["created_at"] = str(data["created_at"])

    return data