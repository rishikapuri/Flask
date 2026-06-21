"""
app.py
Resume Screener — Recruiter POV
Flask web app: upload a job description + resumes, rank candidates
by embedding similarity (TF-IDF + cosine similarity).
"""

import os
import uuid
import random

from flask import Flask, render_template, request, redirect, url_for, flash

from parser import (
    extract_text,
    guess_name,
    extract_email,
    extract_phone,
    clean_filename_as_fallback,
)
from ranker import rank_resumes, score_band

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_DIR = os.path.join(BASE_DIR, "uploads")
ALLOWED_EXTENSIONS = {"pdf", "docx", "txt"}

app = Flask(__name__)
app.secret_key = "resume-screener-dev-key"
app.config["MAX_CONTENT_LENGTH"] = 25 * 1024 * 1024  # 25 MB total upload cap

os.makedirs(UPLOAD_DIR, exist_ok=True)


def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route("/", methods=["GET"])
def index():
    return render_template("index.html", file_no=f"{random.randint(1000, 9999)}")


@app.route("/screen", methods=["POST"])
def screen():
    job_description = request.form.get("job_description", "").strip()
    files = request.files.getlist("resumes")

    if not job_description:
        flash("Please paste a job description before screening candidates.")
        return redirect(url_for("index"))

    valid_files = [f for f in files if f and f.filename and allowed_file(f.filename)]

    if not valid_files:
        flash("Please upload at least one resume in PDF, DOCX, or TXT format.")
        return redirect(url_for("index"))

    candidates = []
    skipped = []

    for f in valid_files:
        safe_id = uuid.uuid4().hex[:8]
        original_name = f.filename
        ext = original_name.rsplit(".", 1)[1].lower()
        temp_path = os.path.join(UPLOAD_DIR, f"{safe_id}.{ext}")
        f.save(temp_path)

        try:
            raw_text = extract_text(temp_path)
            if not raw_text or not raw_text.strip():
                skipped.append(original_name)
                continue

            fallback_name = clean_filename_as_fallback(original_name)
            candidates.append({
                "id": safe_id,
                "filename": original_name,
                "name": guess_name(raw_text, fallback_name),
                "email": extract_email(raw_text),
                "phone": extract_phone(raw_text),
                "text": raw_text,
                "preview": " ".join(raw_text.split())[:280],
            })
        except Exception:
            skipped.append(original_name)
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)

    if not candidates:
        flash("None of the uploaded files could be read. Please check the formats and try again.")
        return redirect(url_for("index"))

    result = rank_resumes(job_description, [c["text"] for c in candidates])

    for i, c in enumerate(candidates):
        c["score"] = result["scores"][i]
        c["band"] = score_band(c["score"])
        c["matched_keywords"] = result["keywords"][i]
        del c["text"]  # no longer needed, keep payload light

    ranked = sorted(candidates, key=lambda c: c["score"], reverse=True)
    for rank, c in enumerate(ranked, start=1):
        c["rank"] = rank

    return render_template(
        "results.html",
        candidates=ranked,
        job_description=job_description,
        jd_top_terms=result["jd_top_terms"],
        skipped=skipped,
        total=len(ranked),
    )


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
