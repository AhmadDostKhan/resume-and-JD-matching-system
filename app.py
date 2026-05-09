"""
Advanced ATS Resume Matcher (Upgraded from v1)
===============================================
Combines:
  • Old project strength: Keyword matching for skills (rule-based works!)
  • NEW: Cross-encoder sentence-level for education & experience
  • NEW: Better acronym expansion
  • NEW: Semantic understanding without losing precision

Models used:
  - Keyword extraction: all-MiniLM-L6-v2 (bi-encoder)
  - Education/Experience: cross-encoder/ms-marco-MiniLM-L-6-v2
  - Skills: Hybrid keyword + semantic

Install:
    pip install flask flask-cors pdfplumber python-docx sentence-transformers nltk scikit-learn numpy

Run:
    python app.py
"""

from flask import Flask, request, render_template, jsonify
from sentence_transformers import SentenceTransformer, CrossEncoder, util
import pdfplumber, docx, re, numpy as np, os, io, math
import nltk
from sklearn.metrics.pairwise import cosine_similarity

try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)
    nltk.download('punkt_tab', quiet=True)

from nltk.tokenize import sent_tokenize

app = Flask(__name__)
from flask_cors import CORS
CORS(app)

# ── MODELS ─────────────────────────────────────────────────────────────────────
print("Loading models…")
BI_ENCODER = SentenceTransformer("all-MiniLM-L6-v2")  # for skills
CROSS_ENCODER = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")  # for edu/exp
print("Models ready.")

# ── DATA ────────────────────────────────────────────────────────────────────────

ACRONYMS = {
    "ml": "machine learning", "ai": "artificial intelligence",
    "nlp": "natural language processing", "dl": "deep learning",
    "cv": "computer vision", "ds": "data science", "api": "application programming interface",
    "rest": "representational state transfer", "sql": "structured query language",
    "ci": "continuous integration", "cd": "continuous deployment",
    "devops": "development operations", "qa": "quality assurance",
}

DEGREE_LEVELS = {
    "bachelor": ["bachelor", "bsc", "bs", "b.sc", "b.s.", "undergraduate"],
    "master": ["master", "msc", "ms", "m.sc", "m.s.", "graduate"],
    "phd": ["phd", "doctorate", "ph.d", "d.phil"],
}

# ── UTILITIES ───────────────────────────────────────────────────────────────────

def clean_text(text):
    """Clean text for processing."""
    text = text.lower()
    text = re.sub(r'[^a-z0-9\s]', ' ', text)
    return re.sub(r'\s+', ' ', text).strip()

def expand_acronyms(text):
    """Expand common acronyms."""
    for acr, full in ACRONYMS.items():
        text = re.sub(rf'\b{acr}\b', full, text.lower())
    return text

def extract_text_from_resume(file):
    """Extract text from PDF, DOCX, TXT."""
    name = file.filename.lower()
    data = file.read()
    
    if name.endswith(".pdf"):
        text = ""
        with pdfplumber.open(io.BytesIO(data)) as pdf:
            for page in pdf.pages:
                text += page.extract_text() or ""
        return text
    elif name.endswith(".docx"):
        doc = docx.Document(io.BytesIO(data))
        return "\n".join(p.text for p in doc.paragraphs if p.text)
    elif name.endswith(".txt"):
        return data.decode("utf-8", errors="ignore")
    return ""

def extract_section(text, keywords):
    """Extract section from resume by keywords."""
    text_lower = text.lower()
    indices = [text_lower.find(kw) for kw in keywords if kw in text_lower]
    if not indices:
        return ""
    start = min(i for i in indices if i != -1)
    # Find next section or end of text
    all_kws = [
        "skill", "experience", "education", "project", "certification",
        "award", "achievement", "language", "interest", "reference"
    ]
    next_start = len(text)
    for kw in all_kws:
        idx = text_lower.find(kw, start + 10)
        if idx != -1:
            next_start = min(next_start, idx)
    return text[start:next_start].strip()

def parse_resume_sections(text):
    """Parse resume into skills, experience, education."""
    return {
        "skills": extract_section(text, ["skill", "technical skill", "competenc"]),
        "experience": extract_section(text, ["experience", "work history", "employment"]),
        "education": extract_section(text, ["education", "academic", "qualification"]),
    }

# ── SCORING FUNCTIONS ──────────────────────────────────────────────────────────

def score_skills_hybrid(jd_skills_str, resume_text):
    """
    Skills matching: Blend keyword matching (70%) + semantic (30%)
    Keeps the strength of your old keyword approach but adds semantic understanding.
    
    Old approach: Pure keyword matching
    New approach: Keywords PLUS semantic backup for paraphrases
    """
    jd_skills = [s.strip() for s in jd_skills_str.split(",") if s.strip()]
    resume_lower = resume_text.lower()
    
    # Part 1: Keyword matching (your old method - it works!)
    keyword_score = 0
    for skill in jd_skills:
        is_core = skill.endswith("*")
        skill_clean = skill.replace("*", "").strip().lower()
        weight = 2 if is_core else 1
        
        # Check if any word of the skill is in resume
        skill_words = re.findall(r'\b\w+\b', skill_clean)
        if any(word in resume_lower for word in skill_words):
            keyword_score += weight
    
    keyword_score = keyword_score / sum(2 if s.endswith("*") else 1 for s in jd_skills) if jd_skills else 0
    
    # Part 2: Semantic backup (for paraphrases)
    jd_skills_text = " ".join(jd_skills)
    emb_jd = BI_ENCODER.encode(jd_skills_text, convert_to_tensor=True, normalize_embeddings=True)
    emb_resume = BI_ENCODER.encode(resume_text[:1200], convert_to_tensor=True, normalize_embeddings=True)
    semantic_score = util.cos_sim(emb_jd, emb_resume).item()
    
    # Blend: keywords are reliable, but semantic catches misses
    final = 0.70 * keyword_score + 0.30 * semantic_score
    return float(max(0.0, min(1.0, final)))

def extract_years(text):
    """Extract years from text."""
    matches = re.findall(r'(\d+)\+?\s*(?:years?|yrs)', text.lower())
    return max(map(int, matches)) if matches else 0

def score_experience_advanced(jd_exp, resume_exp):
    """
    Experience: Years check + sentence-level cross-encoder matching
    
    Old approach: 70% years rule + 30% semantic
    New approach: Years rule + sentence-level F1 scoring
    """
    if not jd_exp.strip() or not resume_exp.strip():
        return None
    
    jd_years = extract_years(jd_exp)
    resume_years = extract_years(resume_exp)
    
    # Years requirement (binary)
    years_score = 1.0 if resume_years >= jd_years else 0.5
    
    # Sentence-level semantic
    j_sents = [s.strip() for s in sent_tokenize(jd_exp) if len(s.strip()) > 20][:10]
    r_sents = [s.strip() for s in sent_tokenize(resume_exp) if len(s.strip()) > 20][:15]
    
    if not j_sents or not r_sents:
        # Fallback to full-text cosine if no good sentences
        emb_j = BI_ENCODER.encode(jd_exp[:800], convert_to_tensor=True, normalize_embeddings=True)
        emb_r = BI_ENCODER.encode(resume_exp[:800], convert_to_tensor=True, normalize_embeddings=True)
        semantic = util.cos_sim(emb_j, emb_r).item()
        return float(0.4 * years_score + 0.6 * semantic)
    
    # Cross-encoder on sentences
    pairs = [(j, r) for j in j_sents for r in r_sents]
    raw_scores = CROSS_ENCODER.predict(pairs)
    
    def sigmoid(x):
        return 1.0 / (1.0 + math.exp(-x))
    
    scores = [sigmoid(float(s)) for s in raw_scores]
    matrix = [scores[i*len(r_sents):(i+1)*len(r_sents)] for i in range(len(j_sents))]
    best_per_jd = [max(row) for row in matrix]
    coverage = sum(best_per_jd) / len(best_per_jd)
    
    # F1 blend
    final = 0.35 * years_score + 0.65 * coverage
    return float(max(0.0, min(1.0, final)))

def score_education_advanced(jd_edu, resume_edu):
    """
    Education: Degree level matching + sentence-level cross-encoder
    
    Old approach: Exact degree level match + semantic
    New approach: Degree level + sentence matching for field/requirements
    """
    if not jd_edu.strip() or not resume_edu.strip():
        return None
    
    # Degree level matching (your old method)
    jd_level = next((lvl for lvl, kws in DEGREE_LEVELS.items() 
                     if any(k in jd_edu.lower() for k in kws)), None)
    resume_level = next((lvl for lvl, kws in DEGREE_LEVELS.items() 
                         if any(k in resume_edu.lower() for k in kws)), None)
    
    # Degree level score
    if jd_level and resume_level:
        level_hierarchy = {"bachelor": 1, "master": 2, "phd": 3}
        jd_h = level_hierarchy.get(jd_level, 0)
        res_h = level_hierarchy.get(resume_level, 0)
        level_score = 1.0 if res_h >= jd_h else max(0.4, (res_h / jd_h))
    else:
        level_score = 0.5
    
    # Sentence-level semantic
    j_sents = [s.strip() for s in sent_tokenize(jd_edu) if len(s.strip()) > 20][:8]
    r_sents = [s.strip() for s in sent_tokenize(resume_edu) if len(s.strip()) > 20][:12]
    
    if not j_sents or not r_sents:
        emb_j = BI_ENCODER.encode(jd_edu[:800], convert_to_tensor=True, normalize_embeddings=True)
        emb_r = BI_ENCODER.encode(resume_edu[:800], convert_to_tensor=True, normalize_embeddings=True)
        semantic = util.cos_sim(emb_j, emb_r).item()
        return float(0.3 * level_score + 0.7 * semantic)
    
    pairs = [(j, r) for j in j_sents for r in r_sents]
    raw_scores = CROSS_ENCODER.predict(pairs)
    
    def sigmoid(x):
        return 1.0 / (1.0 + math.exp(-x))
    
    scores = [sigmoid(float(s)) for s in raw_scores]
    matrix = [scores[i*len(r_sents):(i+1)*len(r_sents)] for i in range(len(j_sents))]
    best_per_jd = [max(row) for row in matrix]
    coverage = sum(best_per_jd) / len(best_per_jd)
    
    final = 0.25 * level_score + 0.75 * coverage
    return float(max(0.0, min(1.0, final)))

# ── MAIN MATCHING FUNCTION ─────────────────────────────────────────────────────

def ats_match_advanced(jd_skills, jd_exp, jd_edu, resume_text):
    """
    Advanced ATS matching combining keyword + semantic understanding.
    
    Scoring breakdown:
      - Skills: 50% (hybrid keyword + semantic)
      - Experience: 30% (years + sentence-level)
      - Education: 20% (degree level + field match)
    """
    resume_expanded = expand_acronyms(resume_text)
    sections = parse_resume_sections(resume_expanded)
    
    skills_score = score_skills_hybrid(jd_skills, sections["skills"] or resume_expanded)
    exp_score = score_experience_advanced(jd_exp, sections["experience"] or resume_expanded)
    edu_score = score_education_advanced(jd_edu, sections["education"] or resume_expanded)
    
    # Handle None values (missing sections)
    if exp_score is None:
        exp_score = 0.0
    if edu_score is None:
        edu_score = 0.0
    
    # Weighted average: Skills dominant (it was your strength!)
    final_score = 0.50 * skills_score + 0.30 * exp_score + 0.20 * edu_score
    
    return {
        "final_score": round(final_score * 100, 2),
        "skills_score": round(skills_score * 100, 2),
        "experience_score": round(exp_score * 100, 2),
        "education_score": round(edu_score * 100, 2),
        "method": "Advanced: Hybrid keyword + cross-encoder sentence-level",
    }

# ── ROUTES ─────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/api/match", methods=["POST"])
def api_match():
    try:
        if "resume" not in request.files:
            return jsonify({"error": "No resume file uploaded"}), 400
        
        skills = request.form.get("skills", "").strip()
        experience = request.form.get("experience", "").strip()
        education = request.form.get("education", "").strip()
        resume_file = request.files["resume"]
        
        if not all([skills, experience, education]):
            return jsonify({"error": "Missing skills, experience, or education"}), 400
        
        resume_text = extract_text_from_resume(resume_file)
        if not resume_text:
            return jsonify({"error": "Could not extract text from resume"}), 400
        
        result = ats_match_advanced(skills, experience, education, resume_text)
        
        return jsonify({
            "final_score": result["final_score"],
            "skills_score": result["skills_score"],
            "experience_score": result["experience_score"],
            "education_score": result["education_score"],
            "method": result["method"],
            "status": "success"
        })
    
    except Exception as e:
        import traceback
        print(traceback.format_exc())
        return jsonify({"error": str(e), "status": "error"}), 500

@app.route("/health")
def health():
    return jsonify({
        "status": "healthy",
        "version": "2.0 - Advanced Semantic",
        "models": {
            "skills": "all-MiniLM-L6-v2 (hybrid keyword + semantic)",
            "experience": "cross-encoder/ms-marco-MiniLM-L-6-v2 (sentence-level)",
            "education": "cross-encoder/ms-marco-MiniLM-L-6-v2 (degree + field)"
        }
    })

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=True, host="0.0.0.0", port=port)
