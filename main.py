import os
import re
import json
from datetime import datetime
from collections import Counter

try:
    import requests
    HAS_REQUESTS = True
except Exception:
    requests = None
    HAS_REQUESTS = False
from flask import Flask, request, render_template, redirect, url_for
from werkzeug.utils import secure_filename

# optional heavy dependencies: handle missing packages gracefully so
# the app can start even if they are not installed in the environment.
try:
    import fitz          # PyMuPDF for PDF
    HAS_FITZ = True
except Exception:
    fitz = None
    HAS_FITZ = False

try:
    import docx          # python-docx for DOCX
    HAS_DOCX = True
except Exception:
    docx = None
    HAS_DOCX = False

try:
    import PyPDF2
    HAS_PYPDF = True
except Exception:
    PyPDF2 = None
    HAS_PYPDF = False

try:
    import pytesseract
    HAS_PYTESSACT = True
except Exception:
    pytesseract = None
    HAS_PYTESSACT = False

try:
    from pdf2image import convert_from_path
    HAS_PDF2IMAGE = True
except Exception:
    convert_from_path = None
    HAS_PDF2IMAGE = False

try:
    from PIL import Image
    HAS_PIL = True
except Exception:
    Image = None
    HAS_PIL = False

try:
    from sentence_transformers import SentenceTransformer, util
    HAS_SENT_TRANS = True
except Exception:
    SentenceTransformer = None
    util = None
    HAS_SENT_TRANS = False

# ----------------------------------------------------
# Flask Setup
# ----------------------------------------------------
app = Flask(__name__)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

UPLOAD_FOLDER = os.path.join(BASE_DIR, "uploads")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

ALLOWED_EXTENSIONS = {"pdf", "docx"}
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

APPLIED_JOBS_FILE = os.path.join(BASE_DIR, "applied_jobs.json")

# ----------------------------------------------------
# Model + Skills Config
# ----------------------------------------------------
model = SentenceTransformer("all-MiniLM-L6-v2") if HAS_SENT_TRANS else None

SKILL_KEYWORDS = [
    "sql", "excel", "tableau", "power bi", "python", "r", "pandas",
    "forecasting", "budgeting", "kpi", "dashboard", "dashboards",
    "machine learning", "deep learning", "statistics", "regression",
    "etl", "data warehouse", "aws", "azure", "gcp",
    "stakeholder", "storytelling", "communication",
    "data visualization", "presentation", "business analysis",
]

CATEGORIES = [
    "All",
    "Data / Analytics",
    "Business / Strategy",
    "Software / Engineering",
    "Cloud / DevOps",
    "Finance / Banking",
    "Operations / Supply Chain",
    "Marketing / Growth",
    "HR / People Analytics",
    "Sales / Revenue",
    "Healthcare / Clinical",
    "Education / Training",
]

CATEGORY_QUERY_MAP = {
    "Data / Analytics": "data analyst OR data scientist OR analytics",
    "Business / Strategy": "business analyst OR strategy analyst",
    "Software / Engineering": "software engineer OR full stack developer OR backend engineer",
    "Cloud / DevOps": "cloud engineer OR devops OR site reliability",
    "Finance / Banking": "financial analyst OR banking analyst OR fp&a",
    "Operations / Supply Chain": "operations analyst OR supply chain analyst",
    "Marketing / Growth": "marketing analyst OR growth analyst",
    "HR / People Analytics": "hr analyst OR people analytics",
    "Sales / Revenue": "sales analyst OR revenue operations",
    "Healthcare / Clinical": "healthcare analyst OR clinical data",
    "Education / Training": "education data analyst OR learning analytics",
}

COUNTRIES = {
    "United States": ("us", "United States"),
    "India": ("in", "India"),
    "United Kingdom": ("gb", "United Kingdom"),
    "Canada": ("ca", "Canada"),
    "Australia": ("au", "Australia"),
    "Germany": ("de", "Germany"),
    "France": ("fr", "France"),
    "Netherlands": ("nl", "Netherlands"),
    "Singapore": ("sg", "Singapore"),
    "Brazil": ("br", "Brazil"),
}

# Simple industry keyword buckets
INDUSTRY_KEYWORDS = {
    "tech": ["software", "developer", "saas", "cloud", "engineer", "it"],
    "finance": ["bank", "financial", "investment", "trading", "fp&a", "fintech"],
    "healthcare": ["hospital", "clinical", "pharma", "healthcare", "medical"],
    "education": ["school", "university", "college", "education", "edtech"],
    "marketing": ["marketing", "brand", "campaign", "seo", "content", "social media"],
    "operations": ["supply chain", "logistics", "warehouse", "operations", "fulfillment"],
}

# ----------------------------------------------------
# Adzuna API Config
# ----------------------------------------------------
ADZUNA_APP_ID = "00f602bf"
ADZUNA_API_KEY = "a190d32c263caa6056651b729cb47fbb"


# ----------------------------------------------------
# Helper functions (generic)
# ----------------------------------------------------
def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def extract_text_from_resume(filepath: str) -> str:
    ext = filepath.rsplit(".", 1)[1].lower()

    if ext == "pdf":
        # Try PyMuPDF first, then PyPDF2 as a fallback
        if HAS_FITZ:
            text = ""
            with fitz.open(filepath) as doc:
                for page in doc:
                    text += page.get_text()
            return text

        if HAS_PYPDF:
            try:
                reader = PyPDF2.PdfReader(filepath)
                text = []
                for p in reader.pages:
                    try:
                        text.append(p.extract_text() or "")
                    except Exception:
                        text.append("")
                return "\n".join(text)
            except Exception:
                return ""

        # No PDF extraction lib available
        # If we have OCR libs, try OCR on PDF pages (use pdf2image when available)
        if HAS_PYTESSACT and HAS_PDF2IMAGE and HAS_PIL:
            try:
                images = convert_from_path(filepath, dpi=200)
                ocr_text = []
                for img in images:
                    try:
                        ocr_text.append(pytesseract.image_to_string(img))
                    except Exception:
                        ocr_text.append("")
                return "\n".join(ocr_text)
            except Exception:
                return ""

        return ""

    if ext == "docx":
        if HAS_DOCX:
            document = docx.Document(filepath)
            return "\n".join(p.text for p in document.paragraphs)

        # Fallback: read the document.xml from the .docx zip and extract text
        try:
            import zipfile
            import xml.etree.ElementTree as ET

            with zipfile.ZipFile(filepath) as z:
                with z.open('word/document.xml') as docxml:
                    tree = ET.parse(docxml)
                    root = tree.getroot()
                    # Word XML uses the w:t tag for text
                    texts = [t.text for t in root.iter() if t.tag.endswith('}t') and t.text]
                    return "\n".join(texts)
        except Exception:
            return ""

    return ""


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text.lower())


def extract_skills_from_text(text: str) -> set:
    text_norm = normalize_text(text)
    found = set()
    for skill in SKILL_KEYWORDS:
        if skill in text_norm:
            found.add(skill)
    return found


def fetch_jobs_from_adzuna(country_code: str,
                           location: str,
                           category: str,
                           max_results: int = 50):
    if not country_code:
        country_code = "us"

    if not HAS_REQUESTS:
        print("[fetch-jobs] requests library not installed; skipping API call")
        return []

    url = f"https://api.adzuna.com/v1/api/jobs/{country_code}/search/1"

    if category and category != "All":
        what_query = CATEGORY_QUERY_MAP.get(category, "")
    else:
        what_query = "analyst OR analytics OR data"

    params = {
        "app_id": ADZUNA_APP_ID,
        "app_key": ADZUNA_API_KEY,
        "results_per_page": max_results,
        "content-type": "application/json",
    }

    if location:
        params["where"] = location

    if what_query:
        params["what"] = what_query

    try:
        resp = requests.get(url, params=params, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        return data.get("results", [])
    except Exception as e:
        print("Error calling Adzuna:", e)
        return []


def load_applied_jobs():
    if not os.path.exists(APPLIED_JOBS_FILE):
        return []
    try:
        with open(APPLIED_JOBS_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except json.JSONDecodeError:
        return []


def save_applied_jobs(jobs):
    with open(APPLIED_JOBS_FILE, "w", encoding="utf-8") as f:
        json.dump(jobs, f, indent=2)


# ----------------------------------------------------
# ATS v2 helper functions (Enterprise-style heuristics)
# ----------------------------------------------------
def estimate_years_experience(text: str) -> int:
    """
    Roughly estimate max 'X years' mentioned in the text.
    E.g., '3+ years', '5 years' etc.
    """
    matches = re.findall(r'(\d+)\+?\s+years', text.lower())
    if not matches:
        return 0
    try:
        return max(int(m) for m in matches)
    except ValueError:
        return 0


def extract_years_timeline(text: str):
    """
    Extract calendar years mentioned in the text.
    Used to approximate career span and gaps.
    """
    years = set()
    for y in re.findall(r'\b(19[7-9]\d|20[0-4]\d)\b', text):
        try:
            years.add(int(y))
        except ValueError:
            continue
    if not years:
        return [], 0
    sorted_years = sorted(years)
    span = sorted_years[-1] - sorted_years[0]
    return sorted_years, span


def estimate_job_hops(text: str) -> int:
    """
    Very rough proxy for number of roles based on date patterns and
    typical separators like ' - ' or ' to ' or line breaks with bullets.
    """
    # count patterns like '2019 - 2021', '2020 to 2023'
    range_matches = re.findall(r'(19[7-9]\d|20[0-4]\d)\s*[-–to]{1,3}\s*(19[7-9]\d|20[0-4]\d)', text)
    hops_from_ranges = len(range_matches)

    # bullets often indicate separate roles
    bullet_lines = sum(1 for line in text.splitlines() if line.strip().startswith(("-", "•", "*")))
    # combine signals
    approx_roles = max(hops_from_ranges, bullet_lines // 3)
    return max(1, approx_roles)


def stability_score(resume_text: str) -> float:
    """
    Higher when career span is reasonable vs job hops.
    If someone changes jobs too often relative to span, score goes down.
    """
    years, span = extract_years_timeline(resume_text)
    if not years or span <= 0:
        return 0.8  # neutral if we can't tell

    roles = estimate_job_hops(resume_text)
    # Ideal: one job every ~3 years
    ideal_roles = max(1, span / 3.0)
    ratio = roles / ideal_roles

    if ratio <= 1.2:
        return 1.0  # stable
    if ratio <= 2.0:
        return 0.8
    if ratio <= 3.0:
        return 0.6
    return 0.4  # lots of hopping


def gap_score(resume_text: str) -> float:
    """
    Penalize large gaps (2+ years) between years on timeline.
    """
    years, span = extract_years_timeline(resume_text)
    if len(years) < 3:
        return 0.9  # we don't have enough info

    gaps = 0
    for i in range(len(years) - 1):
        diff = years[i + 1] - years[i]
        if diff >= 3:
            gaps += 1

    if gaps == 0:
        return 1.0
    if gaps == 1:
        return 0.85
    if gaps == 2:
        return 0.7
    return 0.5


def seniority_level(text: str) -> int:
    """
    0=intern, 1=junior, 2=mid, 3=senior, 4=lead, 5=manager/director.
    """
    t = text.lower()
    level = 2  # mid by default

    if "intern" in t or "internship" in t:
        return 0
    if any(k in t for k in ["junior", " jr ", " jr.", "entry level"]):
        level = 1
    if any(k in t for k in ["senior", " sr ", " sr.", "senior-level"]):
        level = 3
    if any(k in t for k in ["lead", "principal"]):
        level = 4
    if any(k in t for k in ["manager", "director", "head of"]):
        level = 5

    return level


def seniority_match_score(job_text: str, resume_text: str) -> float:
    job_level = seniority_level(job_text)
    resume_level = seniority_level(resume_text)
    diff = abs(job_level - resume_level)
    return max(0.0, 1.0 - diff / 3.0)


def degree_level(text: str) -> int:
    """
    0 = none detected,
    1 = associate / diploma,
    2 = bachelor's,
    3 = master's / mba,
    4 = phd / doctorate
    """
    t = text.lower()
    level = 0

    if any(k in t for k in ["phd", "doctorate", "dphil"]):
        level = max(level, 4)
    if any(k in t for k in ["master", "msc", "m.s.", "m.s ", "mtech", "m.tech", "mba"]):
        level = max(level, 3)
    if any(k in t for k in ["bachelor", "bsc", "b.s.", "b.s ", "btech", "b.tech", "b.e."]):
        level = max(level, 2)
    if any(k in t for k in ["associate", "diploma"]):
        level = max(level, 1)

    return level


def education_match_score(job_text: str, resume_text: str) -> float:
    job_req = degree_level(job_text)
    resume_deg = degree_level(resume_text)

    if job_req == 0:
        return 0.9  # job doesn't care explicitly

    if resume_deg >= job_req:
        return 1.0

    # penalty if resume below required level
    return max(0.4, (resume_deg + 1) / (job_req + 1))


def detect_industries(text: str) -> set:
    t = text.lower()
    found = set()
    for industry, kws in INDUSTRY_KEYWORDS.items():
        if any(k in t for k in kws):
            found.add(industry)
    if not found:
        found.add("generic")
    return found


def industry_match_score(job_text: str, resume_text: str) -> float:
    job_ind = detect_industries(job_text)
    res_ind = detect_industries(resume_text)

    if job_ind & res_ind:
        return 1.0

    # if job is generic or resume is generic → neutral
    if "generic" in job_ind or "generic" in res_ind:
        return 0.85

    return 0.7


def resume_format_score(resume_text: str) -> float:
    """
    Rough estimate of ATS-friendly formatting:
    - presence of common sections
    - presence of bullet points
    """
    t = resume_text.lower()
    score = 0.5

    sections = 0
    for sec in ["experience", "work history", "education", "skills", "projects"]:
        if sec in t:
            sections += 1
    if sections >= 3:
        score += 0.3
    elif sections >= 1:
        score += 0.15

    bullets = sum(1 for line in resume_text.splitlines() if line.strip().startswith(("-", "•", "*")))
    if bullets >= 5:
        score += 0.2
    elif bullets >= 2:
        score += 0.1

    return max(0.5, min(1.0, score))


# ----------------------------------------------------
# Routes
# ----------------------------------------------------
@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "GET":
        return render_template(
            "index.html",
            categories=CATEGORIES,
            countries=list(COUNTRIES.keys()),
        )

    # POST: handle resume + filters
    file = request.files.get("resume")
    selected_category = request.form.get("category", "All")
    selected_country = request.form.get("country", "United States")
    location = (request.form.get("location") or "").strip()

    if not file or not allowed_file(file.filename):
        return "Invalid file type. Only PDF and DOCX allowed."

    country_code, default_where = COUNTRIES.get(
        selected_country,
        ("us", "United States"),
    )

    if not location:
        location = default_where

    filename = secure_filename(file.filename)
    save_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    file.save(save_path)

    resume_text = extract_text_from_resume(save_path)
    if not resume_text.strip():
        # Provide a helpful diagnostic so the user can understand why extraction failed
        try:
            size = os.path.getsize(save_path)
        except Exception:
            size = None

        ext = filename.rsplit(".", 1)[1].lower() if "." in filename else ""

        available = []
        if HAS_FITZ:
            available.append("pymupdf")
        if HAS_PYPDF:
            available.append("pypdf2")
        if HAS_DOCX:
            available.append("python-docx")
        if HAS_PYTESSACT:
            available.append("pytesseract")
        if HAS_PDF2IMAGE:
            available.append("pdf2image")

        msg_lines = [
            "Could not extract text from resume.",
            f"Detected extension: {ext}",
            f"Saved file size: {size if size is not None else 'unknown'} bytes",
            f"Available extractors: {', '.join(available) if available else 'none'}",
            "Possible reasons:",
            " - The file is a scanned image PDF (needs OCR).",
            " - The required libraries are not installed (install pymupdf or python-docx).",
            " - The .docx is password-protected or malformed.",
            "Suggestions:",
            " - Try opening the resume in a text editor to confirm it's not an image-only PDF.",
            " - Install dependencies: pip install pymupdf python-docx PyPDF2 pdf2image pytesseract",
            " - For OCR (image PDFs) also install system packages: macOS: brew install tesseract poppler",
            " - If you want, attach the resume file and I can inspect it.",
        ]

        # log to server console for debugging
        print("[resume-extract-fail]", {"path": save_path, "ext": ext, "size": size, "available": available})

        return "\n".join(msg_lines)

    resume_skills = extract_skills_from_text(resume_text)
    # embeddings may not be available if sentence_transformers is not installed
    resume_emb = model.encode(resume_text, convert_to_tensor=True) if model else None

    # ATS v2: resume-level features
    resume_years = estimate_years_experience(resume_text)
    resume_format = resume_format_score(resume_text)
    resume_stability = stability_score(resume_text)
    resume_gap = gap_score(resume_text)
    resume_industries = detect_industries(resume_text)

    raw_jobs = fetch_jobs_from_adzuna(
        country_code=country_code,
        location=location,
        category=selected_category,
        max_results=50,
    )

    job_results = []
    all_required_skills = set()
    missing_freq = Counter()

    for item in raw_jobs:
        title = (item.get("title") or "").strip()
        company = ((item.get("company") or {}).get("display_name") or "").strip()
        job_location = ((item.get("location") or {}).get("display_name") or "").strip()
        description = (item.get("description") or "").strip()
        url = item.get("redirect_url") or ""

        if not title and not description:
            continue

        clean_desc = re.sub(r"\s+", " ", description)
        short_desc = clean_desc[:350] + "..." if len(clean_desc) > 350 else clean_desc

        job_text = f"{title}. {clean_desc}"
        job_text_norm = normalize_text(job_text)

        job_emb = model.encode(job_text, convert_to_tensor=True) if model else None
        if model and util is not None and resume_emb is not None and job_emb is not None:
            try:
                sim = util.pytorch_cos_sim(resume_emb, job_emb).item()
            except Exception:
                sim = 0.0
        else:
            sim = 0.0
        sim = max(sim, 0.0)

        job_skills = {s for s in SKILL_KEYWORDS if s in job_text_norm}
        all_required_skills.update(job_skills)

        overlap = job_skills & resume_skills
        missing = job_skills - resume_skills
        missing_freq.update(missing)

        skill_match_pct = (len(overlap) / len(job_skills) * 100) if job_skills else 0.0
        skill_component = skill_match_pct / 100.0

        job_years = estimate_years_experience(clean_desc)
        if job_years > 0 and resume_years > 0:
            exp_ratio = resume_years / job_years
            exp_component = max(0.0, min(1.0, exp_ratio))
        else:
            exp_component = 0.85  # neutral/ok if not specified

        seniority_component = seniority_match_score(title + " " + clean_desc, resume_text)
        education_component = education_match_score(clean_desc, resume_text)
        industry_component = industry_match_score(clean_desc, resume_text)

        # combine gap and stability into one "stability" factor
        stability_component = 0.5 * resume_stability + 0.5 * resume_gap

        sim_component = sim

        # ATS v2 Enterprise-style weighting
        ats_raw = (
            0.25 * sim_component +
            0.25 * skill_component +
            0.15 * exp_component +
            0.10 * seniority_component +
            0.10 * education_component +
            0.05 * industry_component +
            0.05 * stability_component +
            0.05 * resume_format
        )

        ats_score = round(100 * max(0.0, min(1.0, ats_raw)), 1)

        job_results.append(
            {
                "title": title,
                "company": company,
                "location": job_location,
                "url": url,
                "similarity": round(sim, 3),
                "skill_match": round(skill_match_pct, 1),
                "ats_score": ats_score,
                "missing_skills": sorted(missing),
                "description": short_desc,
            }
        )

    # Sort: primary ATS, then similarity
    job_results.sort(key=lambda j: (j["ats_score"], j["similarity"]), reverse=True)
    total_jobs = len(job_results)

    job_titles = [j["title"][:30] for j in job_results[:20]]
    sim_scores = [j["similarity"] for j in job_results[:20]]

    covered = len(all_required_skills & resume_skills)
    missing_overall = max(len(all_required_skills) - covered, 0)
    coverage_stats = {
        "covered": covered,
        "missing": missing_overall,
    }

    common_missing_skills = [
        skill for skill, count in missing_freq.most_common(8)
    ]

    return render_template(
        "results.html",
        jobs=job_results,
        total_jobs=total_jobs,
        resume_skills=sorted(resume_skills),
        job_titles=job_titles,
        sim_scores=sim_scores,
        coverage_stats=coverage_stats,
        common_missing_skills=common_missing_skills,
    )


@app.route("/applied-jobs", methods=["GET", "POST"])
def applied_jobs():
    if request.method == "POST":
        title = (request.form.get("title") or "").strip()
        company = (request.form.get("company") or "").strip()
        location = (request.form.get("location") or "").strip()
        url = (request.form.get("url") or "").strip()

        if title and company and url:
            jobs = load_applied_jobs()
            exists = any(
                j["title"] == title and j["company"] == company and j["url"] == url
                for j in jobs
            )
            if not exists:
                jobs.append(
                    {
                        "title": title,
                        "company": company,
                        "location": location,
                        "url": url,
                        "saved_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
                    }
                )
                save_applied_jobs(jobs)

        return redirect(url_for("applied_jobs"))

    jobs = load_applied_jobs()
    return render_template("applied_jobs.html", jobs=jobs)


if __name__ == "__main__":
    app.run(debug=True)
