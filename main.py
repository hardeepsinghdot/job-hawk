import os
import re
import requests

from flask import Flask, request, render_template
from werkzeug.utils import secure_filename

import fitz          # PyMuPDF for PDF
import docx          # python-docx for DOCX
from sentence_transformers import SentenceTransformer, util

# ----------------------------------------------------
# Flask Setup
# ----------------------------------------------------
app = Flask(__name__)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# UNIVERSAL PATH — ALWAYS WORKS ON ANY PC
UPLOAD_FOLDER = os.path.join(BASE_DIR, "uploads")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

ALLOWED_EXTENSIONS = {"pdf", "docx"}
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# ----------------------------------------------------
# Model + Skills Config
# ----------------------------------------------------
model = SentenceTransformer("all-MiniLM-L6-v2")

SKILL_KEYWORDS = [
    "sql", "excel", "tableau", "power bi", "python", "r", "pandas",
    "forecasting", "budgeting", "kpi", "dashboard", "dashboards",
    "machine learning", "deep learning", "statistics", "regression",
    "etl", "data warehouse", "aws", "azure", "gcp",
    "stakeholder", "storytelling", "communication"
]

CATEGORIES = [
    "Data", "Business", "Finance", "Operations",
    "Marketing", "HR", "Technology", "Sales", "Healthcare"
]

CATEGORY_QUERY_MAP = {
    "Data": "data analyst OR data scientist OR analytics",
    "Business": "business analyst",
    "Finance": "financial analyst OR fp&a OR finance",
    "Operations": "operations analyst OR operations",
    "Marketing": "marketing analyst OR growth",
    "HR": "hr analyst OR people analytics",
    "Technology": "software engineer OR data engineer",
    "Sales": "sales analyst OR revenue operations",
    "Healthcare": "healthcare analyst OR clinical data"
}

# ----------------------------------------------------
# Adzuna API Config
# ----------------------------------------------------
ADZUNA_APP_ID = "00f602bf"
ADZUNA_API_KEY = "a190d32c263caa6056651b729cb47fbb"
ADZUNA_COUNTRY = "us"
ADZUNA_WHERE = "United States"

# ----------------------------------------------------
# Helpers
# ----------------------------------------------------
def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

def extract_text_from_resume(filepath: str) -> str:
    ext = filepath.rsplit(".", 1)[1].lower()

    if ext == "pdf":
        text = ""
        with fitz.open(filepath) as doc:
            for page in doc:
                text += page.get_text()
        return text

    if ext == "docx":
        document = docx.Document(filepath)
        return "\n".join(p.text for p in document.paragraphs)

    return ""

def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text.lower())

def extract_skills_from_text(text: str) -> set:
    found = set()
    text_norm = normalize_text(text)
    for skill in SKILL_KEYWORDS:
        if skill in text_norm:
            found.add(skill)
    return found

def fetch_jobs_from_adzuna(category: str, max_results: int = 50):
    url = f"https://api.adzuna.com/v1/api/jobs/{ADZUNA_COUNTRY}/search/1"

    what_query = CATEGORY_QUERY_MAP.get(category, "")
    params = {
        "app_id": ADZUNA_APP_ID,
        "app_key": ADZUNA_API_KEY,
        "results_per_page": max_results,
        "where": ADZUNA_WHERE,
        "content-type": "application/json",
    }
    if what_query:
        params["what"] = what_query

    try:
        resp = requests.get(url, params=params, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        return data.get("results", [])
    except:
        return []

# ----------------------------------------------------
# Routes
# ----------------------------------------------------
@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "GET":
        return render_template("index.html", categories=CATEGORIES)

    file = request.files.get("resume")
    selected_category = request.form.get("category", "All")

    if not file or not allowed_file(file.filename):
        return "Invalid file type. Only PDF and DOCX allowed."

    filename = secure_filename(file.filename)
    save_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    file.save(save_path)

    resume_text = extract_text_from_resume(save_path)
    if not resume_text.strip():
        return "Could not extract text from resume."

    resume_skills = extract_skills_from_text(resume_text)
    resume_emb = model.encode(resume_text, convert_to_tensor=True)

    raw_jobs = fetch_jobs_from_adzuna(
        selected_category if selected_category != "All" else "",
        max_results=50
    )

    job_results = []
    all_required_skills = set()

    for item in raw_jobs:
        title = (item.get("title") or "").strip()
        company = ((item.get("company") or {}).get("display_name") or "").strip()
        location = ((item.get("location") or {}).get("display_name") or "").strip()
        description = (item.get("description") or "").strip()
        url = item.get("redirect_url") or ""

        if not title and not description:
            continue

        job_text = f"{title}. {description}"
        job_text_norm = normalize_text(job_text)

        job_emb = model.encode(job_text, convert_to_tensor=True)
        sim = util.pytorch_cos_sim(resume_emb, job_emb).item()
        sim = max(sim, 0.0)

        job_skills = {s for s in SKILL_KEYWORDS if s in job_text_norm}
        all_required_skills.update(job_skills)

        overlap = job_skills & resume_skills
        missing = job_skills - resume_skills

        skill_match_pct = (len(overlap) / len(job_skills) * 100) if job_skills else 0
        ats_score = round(100 * (0.6 * sim + 0.4 * (skill_match_pct / 100)), 1)

        job_results.append({
            "title": title,
            "company": company,
            "location": location,
            "url": url,
            "similarity": round(sim, 3),
            "skill_match": round(skill_match_pct, 1),
            "ats_score": ats_score,
            "missing_skills": sorted(missing),
        })

    job_results.sort(key=lambda j: j["similarity"], reverse=True)

    job_titles = [j["title"][:30] for j in job_results[:20]]
    sim_scores = [j["similarity"] for j in job_results[:20]]

    covered = len(all_required_skills & resume_skills)
    missing_overall = max(len(all_required_skills) - covered, 0)

    return render_template(
        "results.html",
        jobs=job_results,
        total_jobs=len(job_results),
        resume_skills=sorted(resume_skills),
        job_titles=job_titles,
        sim_scores=sim_scores,
        coverage_stats={"covered": covered, "missing": missing_overall},
    )

if __name__ == "__main__":
    app.run(debug=True)











