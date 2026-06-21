# Resume Screener — Recruiter POV

Upload a job description and a stack of resumes. The app reads every
resume, converts the job description and each candidate into comparable
vectors, scores them by similarity, and returns a ranked list — fastest
first pass a recruiter can run before opening a single resume by hand.

Pure Python. No external API calls, no internet connection required at
runtime, nothing leaves your machine.

## How the matching works

This uses **TF-IDF vectorization + cosine similarity** as the embedding
and similarity-matching layer:

1. **Text extraction** — `pdfplumber` reads PDFs, `python-docx` reads
   Word docs, plain text is read directly.
2. **Embedding** — the job description and every resume are converted
   into weighted vectors using `TfidfVectorizer` (unigrams + bigrams,
   English stop words removed). Each dimension is a term; each value
   reflects how distinctive that term is to the document.
3. **Similarity matching** — `cosine_similarity` measures the angle
   between the job description's vector and each resume's vector. This
   is the exact same comparison neural sentence-embedding models use —
   the difference here is the vectorizer is classical TF-IDF rather
   than a trained neural network, which keeps the whole thing dependency
   -light, fast, fully explainable, and runnable with no internet access
   and no model downloads.
4. **Score + explanation** — similarity is rescaled to a 0–100 "match
   score" and each candidate's top overlapping terms with the role are
   surfaced, so a recruiter can see *why* someone ranked where they did,
   not just trust a number.

If you want to swap in true neural embeddings later (e.g. OpenAI/Anthropic
embeddings or a local `sentence-transformers` model), only `ranker.py`
needs to change — `rank_resumes()` is the single integration point.

## Setup

```bash
pip install -r requirements.txt
```

## Run

```bash
python app.py
```

Then open **http://localhost:5000** in your browser.

## Project structure

```
resume_screener/
├── app.py              Flask routes (upload, screen, results)
├── parser.py            Resume text extraction (PDF / DOCX / TXT) + field parsing
├── ranker.py             TF-IDF embedding + cosine similarity ranking engine
├── requirements.txt
├── static/
│   └── style.css        UI styling
├── templates/
│   ├── index.html        Upload screen
│   └── results.html      Ranked results screen
└── uploads/              Temporary scratch space (files deleted after each read)
```

## Notes on privacy

Uploaded files are written to `uploads/` only long enough to extract text,
then deleted immediately (see the `finally` block in `app.py`'s `/screen`
route). No resume content is persisted to disk, a database, or any
external service.

## Extending this

- **Bulk/CSV export** — add a route that takes the same ranked output and
  writes it to a CSV or XLSX for ATS import.
- **Weighted scoring** — `ranker.py` could be extended to weight specific
  must-have skills more heavily than general overlap.
- **Real embeddings** — swap `TfidfVectorizer` for an API call to an
  embeddings endpoint if internet access and higher semantic nuance are
  priorities over offline operation.
