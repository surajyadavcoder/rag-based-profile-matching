"""
resume_rag.py
=============
RAG-Based Resume Processing Pipeline — Milestone 2
Part A: Document Processing + Vector Database

Pipeline:
  Load Resumes → Chunk by Section → Generate Embeddings → Store in ChromaDB
"""

import os
import re
import json
import math
import time
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field, asdict


# ── ChromaDB (vector store) ───────────────────────────────────────────────────
try:
    import chromadb
    from chromadb.config import Settings
    CHROMA_AVAILABLE = True
except ImportError:
    CHROMA_AVAILABLE = False
    print("[WARN] ChromaDB not available. Install: pip install chromadb")


# ── Embedding Model (TF-IDF fallback, no external API needed) ────────────────
class TFIDFEmbedder:
    """
    Lightweight TF-IDF based embedder.
    In production, replace with:
      - HuggingFace: sentence-transformers (all-MiniLM-L6-v2)
      - OpenAI: text-embedding-3-small
      - Cohere: embed-english-v3.0
    """

    def __init__(self, dim: int = 512):
        self.dim = dim
        self.vocab: Dict[str, int] = {}
        self.idf: Dict[str, float] = {}
        self.fitted = False

    def _tokenize(self, text: str) -> List[str]:
        tokens = re.findall(r'\b[a-zA-Z][a-zA-Z0-9+#.]*\b', text.lower())
        return [t for t in tokens if len(t) > 1]

    def fit(self, documents: List[str]):
        """Compute IDF over corpus."""
        N = len(documents)
        df: Dict[str, int] = {}
        for doc in documents:
            for token in set(self._tokenize(doc)):
                df[token] = df.get(token, 0) + 1

        # Build vocab from most common terms
        sorted_terms = sorted(df.keys(), key=lambda t: df[t], reverse=True)
        # Use a larger, mostly-collision-free vocab for small corpora
        vocab_terms = sorted_terms[:max(self.dim * 8, 2000)]
        self.vocab = {term: i % self.dim for i, term in enumerate(vocab_terms)}
        self.idf = {t: math.log((N + 1) / (df[t] + 1)) + 1 for t in self.vocab}
        self.fitted = True

    def encode(self, text: str) -> List[float]:
        """Generate TF-IDF embedding vector."""
        if not self.fitted:
            self.fit([text])

        tokens = self._tokenize(text)
        tf: Dict[str, float] = {}
        for t in tokens:
            tf[t] = tf.get(t, 0) + 1
        total = max(len(tokens), 1)

        vec = [0.0] * self.dim
        for token, count in tf.items():
            if token in self.vocab:
                idx = self.vocab[token]
                tfidf = (count / total) * self.idf.get(token, 1.0)
                vec[idx] += tfidf

        # L2 normalize
        norm = math.sqrt(sum(v * v for v in vec)) or 1.0
        return [v / norm for v in vec]

    def encode_batch(self, texts: List[str]) -> List[List[float]]:
        return [self.encode(t) for t in texts]


# ── Data Classes ──────────────────────────────────────────────────────────────

@dataclass
class ResumeChunk:
    """A section-level chunk of a resume with metadata."""
    chunk_id: str
    resume_id: str
    candidate_name: str
    section: str          # SUMMARY, SKILLS, EXPERIENCE, EDUCATION, PROJECTS
    content: str
    embedding: List[float] = field(default_factory=list)


@dataclass
class ResumeMetadata:
    """Extracted metadata for filtering."""
    resume_id: str
    candidate_name: str
    email: str
    skills: List[str]
    experience_years: float
    education: str
    current_role: str
    file_path: str


# ── Document Processing ───────────────────────────────────────────────────────

class ResumeProcessor:
    """Loads, chunks, and extracts metadata from resume files."""

    SECTIONS = ["SUMMARY", "SKILLS", "EXPERIENCE", "EDUCATION", "PROJECTS",
                "CERTIFICATIONS", "ACHIEVEMENTS", "PUBLICATIONS"]

    def load_resume(self, file_path: str) -> str:
        """Milestone 1 file system tool: read resume content."""
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            return f.read()

    def list_resumes(self, directory: str) -> List[str]:
        """Milestone 1 file system tool: list resume files."""
        if not os.path.exists(directory):
            return []
        extensions = {'.txt', '.pdf', '.docx', '.md'}
        files = []
        for fname in os.listdir(directory):
            if any(fname.lower().endswith(ext) for ext in extensions):
                files.append(os.path.join(directory, fname))
        return sorted(files)

    def chunk_by_section(self, text: str, resume_id: str, name: str) -> List[ResumeChunk]:
        """
        Intelligently chunk resume by section headers.
        Preserves semantic coherence of Education, Experience, Skills sections.
        """
        chunks = []
        
        # Build regex pattern for section headers
        section_pattern = '|'.join(self.SECTIONS)
        splits = re.split(
            rf'\n\s*({section_pattern})\s*\n',
            text, flags=re.IGNORECASE
        )

        # First part before any section header (usually name/contact) — skip if too short to be meaningful
        if splits[0].strip() and len(splits[0].strip()) > 60:
            chunks.append(ResumeChunk(
                chunk_id=f"{resume_id}_contact",
                resume_id=resume_id,
                candidate_name=name,
                section="CONTACT",
                content=splits[0].strip()
            ))

        # Parse section pairs
        i = 1
        while i < len(splits) - 1:
            section_name = splits[i].upper()
            section_content = splits[i + 1].strip() if i + 1 < len(splits) else ""
            if section_content:
                chunks.append(ResumeChunk(
                    chunk_id=f"{resume_id}_{section_name.lower()}",
                    resume_id=resume_id,
                    candidate_name=name,
                    section=section_name,
                    content=section_content
                ))
            i += 2

        # Fallback: if no sections found, treat whole text as one chunk
        if len(chunks) <= 1:
            chunks = [ResumeChunk(
                chunk_id=f"{resume_id}_full",
                resume_id=resume_id,
                candidate_name=name,
                section="FULL",
                content=text.strip()
            )]

        return chunks

    def extract_metadata(self, text: str, file_path: str) -> ResumeMetadata:
        """Extract structured metadata for filtering."""
        resume_id = os.path.splitext(os.path.basename(file_path))[0]

        # Name
        name = "Unknown"
        name_match = re.search(r'Name:\s*(.+)', text, re.IGNORECASE)
        if name_match:
            name = name_match.group(1).strip()

        # Email
        email_match = re.search(r'[\w.+-]+@[\w-]+\.[\w.]+', text)
        email = email_match.group(0) if email_match else ""

        # Skills
        tech_pattern = re.compile(
            r'\b(Python|Java|JavaScript|TypeScript|Go|Rust|C\+\+|Scala|R|SQL|'
            r'React|Node\.js|FastAPI|Django|Flask|Spring|Angular|Vue\.js|Next\.js|'
            r'TensorFlow|PyTorch|scikit-learn|XGBoost|LightGBM|HuggingFace|BERT|'
            r'AWS|GCP|Azure|Docker|Kubernetes|Terraform|Airflow|Kafka|Spark|'
            r'PostgreSQL|MongoDB|Redis|MySQL|Cassandra|Snowflake|BigQuery|ChromaDB|Pinecone|'
            r'MLflow|Kubeflow|LangChain|LlamaIndex|FAISS|RAG|LLM|GPT|'
            r'Prometheus|Grafana|Ansible|Jenkins|ArgoCD|Helm)\b',
            re.IGNORECASE
        )
        skills = list(dict.fromkeys(m.group(0) for m in tech_pattern.finditer(text)))

        # Experience years — prefer explicit "N years experience" summary statement,
        # otherwise SUM the per-role "(N years)" durations (not max, since multiple roles add up)
        exp_years = 0.0
        summary_match = re.search(
            r'(\d+(?:\.\d+)?)\+?\s*years?\s+(?:of\s+)?experience',
            text, re.IGNORECASE
        )
        if summary_match:
            val = float(summary_match.group(1))
            if val < 30:
                exp_years = val
        else:
            role_durations = re.findall(r'\((\d+(?:\.\d+)?)\s*years?\)', text, re.IGNORECASE)
            exp_years = sum(float(v) for v in role_durations if float(v) < 30)

        # Education
        edu_match = re.search(r'(B\.Tech|B\.E|M\.Tech|M\.Sc|MBA|MCA|BCA|PhD|B\.Sc|B\.Des)[^\n]*', text, re.IGNORECASE)
        education = edu_match.group(0).strip() if edu_match else ""

        # Current role — match the first "Title | Company | dates" line right after EXPERIENCE
        current_role = ""
        exp_section = re.search(r'EXPERIENCE\s*\n(.+)', text, re.IGNORECASE)
        if exp_section:
            first_line = exp_section.group(1).split('\n')[0]
            role_match = re.match(r'([\w\s/&+]+?)\s*\|', first_line)
            if role_match:
                current_role = role_match.group(1).strip()

        return ResumeMetadata(
            resume_id=resume_id,
            candidate_name=name,
            email=email,
            skills=skills,
            experience_years=exp_years,
            education=education,
            current_role=current_role,
            file_path=file_path
        )


# ── Vector Store (ChromaDB wrapper) ──────────────────────────────────────────

class ResumeVectorStore:
    """
    ChromaDB-based vector store for resume chunks.
    Supports semantic search + metadata filtering.
    """

    def __init__(self, persist_dir: str = "vector_store", embedder: TFIDFEmbedder = None):
        self.persist_dir = persist_dir
        self.embedder = embedder or TFIDFEmbedder(dim=512)
        self.metadata_store: Dict[str, dict] = {}  # resume_id → metadata

        if CHROMA_AVAILABLE:
            self.client = chromadb.PersistentClient(path=persist_dir)
            self.collection = self.client.get_or_create_collection(
                name="resumes",
                metadata={"hnsw:space": "cosine"}
            )
        else:
            # Fallback: in-memory store
            self.collection = None
            self._docs: List[dict] = []

    def index_chunks(self, chunks: List[ResumeChunk], metadata: ResumeMetadata):
        """Add resume chunks to the vector store."""
        if not chunks:
            return

        # Fit embedder on new documents
        texts = [c.content for c in chunks]
        self.embedder.fit(texts + [c.content for c in chunks])

        # Generate embeddings
        embeddings = self.embedder.encode_batch(texts)

        # Store metadata
        self.metadata_store[metadata.resume_id] = asdict(metadata)

        if CHROMA_AVAILABLE and self.collection:
            ids = [c.chunk_id for c in chunks]
            docs = [c.content for c in chunks]
            metas = [{
                "resume_id": c.resume_id,
                "candidate_name": c.candidate_name,
                "section": c.section,
                "experience_years": metadata.experience_years,
                "skills_str": ",".join(metadata.skills[:20]),
                "education": metadata.education,
            } for c in chunks]

            # Upsert (add or update)
            self.collection.upsert(
                ids=ids,
                documents=docs,
                embeddings=embeddings,
                metadatas=metas
            )
        else:
            # Fallback in-memory
            for chunk, emb in zip(chunks, embeddings):
                self._docs.append({
                    "chunk": chunk,
                    "embedding": emb,
                    "metadata": asdict(metadata)
                })

    def search(self, query: str, top_k: int = 10, filters: dict = None) -> List[dict]:
        """
        Semantic search: find top-K resume chunks matching query.
        filters: {"min_experience": 3, "required_skills": ["Python", "AWS"]}
        """
        query_embedding = self.embedder.encode(query)

        if CHROMA_AVAILABLE and self.collection:
            # Build ChromaDB where clause
            where = {}
            if filters:
                if "min_experience" in filters:
                    where["experience_years"] = {"$gte": filters["min_experience"]}

            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=min(top_k, max(self.collection.count(), 1)),
                where=where if where else None,
                include=["documents", "metadatas", "distances"]
            )

            hits = []
            for doc, meta, dist in zip(
                results["documents"][0],
                results["metadatas"][0],
                results["distances"][0]
            ):
                score = 1 - dist  # cosine distance → similarity
                # Skill filter (post-filter)
                if filters and "required_skills" in filters:
                    skills_str = meta.get("skills_str", "").lower()
                    required = [s.lower() for s in filters["required_skills"]]
                    if not any(s in skills_str for s in required):
                        continue
                hits.append({
                    "resume_id": meta["resume_id"],
                    "candidate_name": meta["candidate_name"],
                    "section": meta["section"],
                    "content": doc,
                    "semantic_score": round(score * 100, 2),
                    "experience_years": meta.get("experience_years", 0),
                    "metadata": self.metadata_store.get(meta["resume_id"], {})
                })
            return hits[:top_k]

        else:
            # Fallback: cosine similarity in memory
            def cosine(a, b):
                dot = sum(x * y for x, y in zip(a, b))
                na = math.sqrt(sum(x * x for x in a))
                nb = math.sqrt(sum(y * y for y in b))
                return dot / (na * nb + 1e-9)

            scored = []
            for item in self._docs:
                score = cosine(query_embedding, item["embedding"])
                scored.append({**item, "semantic_score": round(score * 100, 2)})

            scored.sort(key=lambda x: x["semantic_score"], reverse=True)

            # Dedup by resume_id, keeping best chunk per resume
            seen = {}
            for item in scored:
                rid = item["chunk"].resume_id
                if rid not in seen:
                    seen[rid] = item

            hits = []
            for rid, item in list(seen.items())[:top_k]:
                hits.append({
                    "resume_id": item["chunk"].resume_id,
                    "candidate_name": item["chunk"].candidate_name,
                    "section": item["chunk"].section,
                    "content": item["chunk"].content,
                    "semantic_score": item["semantic_score"],
                    "experience_years": item["metadata"].get("experience_years", 0),
                    "metadata": item["metadata"]
                })
            return hits

    def count(self) -> int:
        if CHROMA_AVAILABLE and self.collection:
            return self.collection.count()
        return len(self._docs)


# ── RAG Pipeline ─────────────────────────────────────────────────────────────

class ResumeRAGPipeline:
    """
    End-to-end RAG pipeline:
    Load → Chunk → Embed → Index → Search
    """

    def __init__(self, resume_dir: str = "resumes", persist_dir: str = "vector_store"):
        self.resume_dir = resume_dir
        self.processor = ResumeProcessor()
        self.embedder = TFIDFEmbedder(dim=512)
        self.vector_store = ResumeVectorStore(persist_dir=persist_dir, embedder=self.embedder)
        self.all_metadata: Dict[str, ResumeMetadata] = {}
        self._indexed = False

    def build_index(self, force_rebuild: bool = False) -> dict:
        """
        Full pipeline: load all resumes → chunk → embed → index.
        Returns build stats.
        """
        files = self.processor.list_resumes(self.resume_dir)
        if not files:
            return {"error": f"No resume files found in {self.resume_dir}"}

        print(f"📂 Found {len(files)} resume files")

        # First pass: collect all text for fitting embedder
        all_texts = []
        file_texts = {}
        for fpath in files:
            text = self.processor.load_resume(fpath)
            file_texts[fpath] = text
            all_texts.append(text)

        # Fit embedder on entire corpus (TF-IDF needs full corpus)
        print("🔧 Fitting embedding model on corpus...")
        self.embedder.fit(all_texts)

        # Second pass: chunk and index
        total_chunks = 0
        indexed_resumes = []

        for fpath in files:
            text = file_texts[fpath]
            resume_id = os.path.splitext(os.path.basename(fpath))[0]

            # Extract metadata
            metadata = self.processor.extract_metadata(text, fpath)
            self.all_metadata[resume_id] = metadata

            # Chunk by section
            chunks = self.processor.chunk_by_section(text, resume_id, metadata.candidate_name)

            # Index into vector store
            self.vector_store.index_chunks(chunks, metadata)

            total_chunks += len(chunks)
            indexed_resumes.append(metadata.candidate_name)
            print(f"  ✓ Indexed: {metadata.candidate_name} ({len(chunks)} chunks)")

        self._indexed = True
        stats = {
            "resumes_indexed": len(files),
            "total_chunks": total_chunks,
            "avg_chunks_per_resume": round(total_chunks / len(files), 1),
            "vector_store_size": self.vector_store.count(),
            "indexed_candidates": indexed_resumes
        }
        print(f"\n✅ Index built: {stats['resumes_indexed']} resumes, {stats['total_chunks']} chunks")
        return stats

    def search(self, query: str, top_k: int = 10, filters: dict = None) -> List[dict]:
        """Semantic search over indexed resumes."""
        if not self._indexed:
            self.build_index()
        return self.vector_store.search(query, top_k=top_k, filters=filters)

    def get_all_metadata(self) -> Dict[str, ResumeMetadata]:
        return self.all_metadata


# ── Main: build and demo ──────────────────────────────────────────────────────

if __name__ == "__main__":
    pipeline = ResumeRAGPipeline(resume_dir="resumes", persist_dir="vector_store")

    print("=" * 60)
    print("  RAG Resume Pipeline — Building Index")
    print("=" * 60)
    stats = pipeline.build_index()

    print(f"\n📊 Build Stats:")
    for k, v in stats.items():
        if k != "indexed_candidates":
            print(f"  {k}: {v}")

    print("\n" + "=" * 60)
    print("  Demo Search: 'Machine Learning Engineer with Python'")
    print("=" * 60)
    results = pipeline.search("Machine Learning Engineer with Python and deep learning", top_k=5)
    for i, r in enumerate(results, 1):
        print(f"  {i}. {r['candidate_name']} [{r['section']}] — Score: {r['semantic_score']}")
