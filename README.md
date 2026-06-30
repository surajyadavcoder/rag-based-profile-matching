#  RAG-Based Profile Matching System
---

##  Overview

This project implements a complete **RAG (Retrieval-Augmented Generation) pipeline** for matching resumes against job descriptions:

- **Part A** (`resume_rag.py`) — Loads resumes, chunks them by semantic section, generates embeddings, and stores them in **ChromaDB**
- **Part B** (`job_matcher.py`) — Parses job descriptions, runs semantic retrieval, and combines it with keyword scoring for accurate hybrid ranking

**Result: 100% top-1 retrieval accuracy** across 5 test job descriptions (see `notebooks/rag_experimentation.ipynb`).

---

##  Architecture

```
┌─────────────────┐
│  31 Resumes      │  resumes/*.txt
└────────┬─────────┘
         │ list_resumes() / read_resume()   ← Milestone 1 file tools
         ▼
┌─────────────────────────┐
│  Section-Based Chunking │  SUMMARY | SKILLS | EXPERIENCE | EDUCATION | PROJECTS
└────────┬─────────────────┘
         │
         ▼
┌─────────────────────────┐
│  TF-IDF Embedding (512d) │  (swap-in ready for OpenAI/Cohere/HuggingFace)
└────────┬─────────────────┘
         │
         ▼
┌─────────────────────────┐
│  ChromaDB Vector Store   │  cosine similarity, persisted to disk
└────────┬─────────────────┘
         │
         ▼
┌──────────────────────────────────┐
│  Job Matcher (Hybrid Scoring)     │
│  0.5×semantic + 0.4×keyword       │
│  + 0.1×experience                 │
└────────┬───────────────────────────┘
         │
         ▼
   Ranked JSON output (top-K matches)
```

---

##  Project Structure

```
rag_profile_matching/
├── resume_rag.py              ← Part A: Document processing + vector DB
├── job_matcher.py             ← Part B: Hybrid job matching engine
├── requirements.txt
├── README.md
├── resumes/                   ← 31 diverse candidate resumes
├── job_descriptions/          ← 5 job descriptions (ML, Full-stack, Data Sci, DevOps, AI)
├── vector_store/              ← ChromaDB persistent storage (auto-generated)
├── reports/                   ← Output: match_JD00X.json files
└── notebooks/
    └── rag_experimentation.ipynb   ← Experiments, accuracy & latency analysis
```

---

##  Part A — Document Processing Pipeline

### Chunking Strategy
Resumes are split by **section headers** (not fixed character counts), so each chunk
stays semantically coherent:

```python
SECTIONS = ["SUMMARY", "SKILLS", "EXPERIENCE", "EDUCATION", "PROJECTS",
            "CERTIFICATIONS", "ACHIEVEMENTS", "PUBLICATIONS"]
```

This avoids the classic RAG failure mode where a fixed-size chunk splits a sentence
or a skill list mid-way, losing context. (Comparison with naive fixed-size chunking
is demonstrated in the notebook, Section 6.)

### Embedding Model
A custom **TF-IDF embedder** (512-dim, L2-normalized) is used by default — no external
API key required, so the project runs out of the box. It's a drop-in interface:

```python
class TFIDFEmbedder:
    def encode(self, text: str) -> List[float]: ...
    def encode_batch(self, texts: List[str]) -> List[List[float]]: ...
```

To upgrade to production-grade embeddings, swap this class for:
```python
from sentence_transformers import SentenceTransformer
model = SentenceTransformer('all-MiniLM-L6-v2')
embedding = model.encode(text).tolist()
```
or call OpenAI's `text-embedding-3-small` / Cohere's `embed-english-v3.0` APIs.

### Vector Database — ChromaDB
Chunks are stored in a **persistent ChromaDB collection** with metadata for filtering:

```python
collection.upsert(
    ids=chunk_ids,
    documents=chunk_texts,
    embeddings=embeddings,
    metadatas=[{"resume_id":..., "experience_years":..., "skills_str":...}]
)
```

### Metadata Extraction
For each resume, the pipeline extracts:

| Field | Method |
|-------|--------|
| Name | Regex on `Name:` line |
| Email | Regex pattern match |
| Skills | Tech-keyword dictionary matching (40+ technologies) |
| Experience Years | Parses explicit "N years experience" statement, or sums per-role durations |
| Education | Degree pattern matching (B.Tech, M.Tech, MBA, etc.) |
| Current Role | First "Title \| Company \| dates" line under EXPERIENCE |

---

##  Part B — Job Matching Engine

### Hybrid Scoring Formula

```
final_score = 0.50 × semantic_score   (ChromaDB cosine similarity)
            + 0.40 × keyword_score    (must-have / nice-to-have overlap)
            + 0.10 × experience_score (years match vs requirement)
```

Keyword scoring is weighted heavily because critical, non-negotiable skills
(e.g. "5+ years Python") need hard matching — pure semantic similarity alone
can rank a vaguely-related profile above a perfectly-matching one.

### Must-Have Filtering
Candidates matching fewer than 20% of a JD's must-have keywords are penalized
(score × 0.5) — they still appear in results (for transparency) but rank low.

### Output Format
Matches the assignment spec exactly:

```json
{
  "job_description": "Full Stack Developer (React + Node.js)",
  "company": "ProductFirst Technologies",
  "top_matches": [
    {
      "rank": 1,
      "candidate_name": "Bob Martinez",
      "resume_path": "resumes/resume_02_bob_martinez.txt",
      "match_score": 61.3,
      "score_breakdown": {
        "semantic_score": 24.2,
        "keyword_score": 100.0,
        "candidate_experience_years": 4.0
      },
      "matched_skills": ["React", "Node.js", "AWS", "TypeScript", "Next.js"],
      "missing_skills": ["Experience with microservices and REST APIs"],
      "relevant_excerpts": ["Full Stack Developer with 4 years experience..."],
      "reasoning": " 4 years experience meets requirement of 3+ | ...",
      "hire_recommendation": "BORDERLINE — Good potential but gaps: ..."
    }
  ]
}
```

---

##  Setup & Run

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Build the RAG index (Part A)
```bash
python resume_rag.py
```
This indexes all 31 resumes into ChromaDB and runs a demo search.

### 3. Run job matching (Part B)
```bash
python job_matcher.py
```
This matches all 5 job descriptions against the indexed resumes and saves
results to `reports/match_JD00X.json`.

### 4. Explore the notebook
```bash
jupyter notebook notebooks/rag_experimentation.ipynb
```

---

##  Performance Metrics

| Metric | Result |
|--------|--------|
| Resumes indexed | 31 |
| Total chunks | ~130 (avg 4.2/resume) |
| Job descriptions tested | 5 |
| **Top-1 retrieval accuracy** | **100% (5/5)** |
| Mean retrieval latency | ~15–20ms |
| Vector store | ChromaDB, persistent, cosine similarity |

### Validated Matches
| Job Description | Top Match | Why |
|---|---|---|
| Senior ML Engineer | Alice Johnson | TensorFlow/PyTorch, 6yr exp, IIT Bombay M.Tech |
| Full Stack Developer | Bob Martinez | React/Node.js/TypeScript, 4yr exp |
| Senior Data Scientist | Alice Johnson | Strong stats/ML background (cross-domain) |
| DevOps Engineer | David Kumar | Kubernetes/Terraform/AWS, 5yr exp, CKA certified |
| AI/LLM Engineer | Wendy Bose | LangChain/RAG/Vector DB experience |

---

##  Production Upgrades

1. **Real embeddings** — Swap `TFIDFEmbedder` for `sentence-transformers` or OpenAI/Cohere APIs
2. **Re-ranking** — Add a cross-encoder (e.g. `ms-marco-MiniLM`) to re-rank top-20 → top-10
3. **PDF support** — Add `pdfplumber` to `ResumeProcessor.load_resume()`
4. **Scale** — Test with 1000+ resumes; benchmark recall@K and precision@K
5. **Async indexing** — Use ChromaDB async client for concurrent resume ingestion

---

##  Author

**Suraj Yadav** | GitHub: https://github.com/surajyadavcoder | Email:Surajyadavx.in@gmail.com