"""
job_matcher.py
==============
Part B: Semantic Job Matching Engine

Accepts a job description → Retrieves top-K candidates via RAG →
Hybrid scoring (semantic + keyword) → Returns ranked JSON output
"""

import re
import json
import time
import math
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field

from resume_rag import ResumeRAGPipeline, ResumeMetadata


# ── Data Classes ──────────────────────────────────────────────────────────────

@dataclass
class JobDescription:
    jd_id: str
    title: str
    company: str
    must_have: List[str]
    nice_to_have: List[str]
    min_experience: float
    raw_text: str


@dataclass
class CandidateMatch:
    candidate_name: str
    resume_path: str
    match_score: float
    semantic_score: float
    keyword_score: float
    experience_years: float
    matched_skills: List[str]
    missing_skills: List[str]
    relevant_excerpts: List[str]
    reasoning: str
    hire_recommendation: str


# ── JD Parser ────────────────────────────────────────────────────────────────

def parse_job_description(jd_text: str, jd_id: str = "JD001") -> JobDescription:
    """Parse raw JD text into structured JobDescription."""

    # Title
    title_match = re.search(r'Position:\s*(.+)', jd_text, re.IGNORECASE)
    title = title_match.group(1).strip() if title_match else "Unknown Role"

    # Company
    company_match = re.search(r'Company:\s*(.+)', jd_text, re.IGNORECASE)
    company = company_match.group(1).strip() if company_match else ""

    # Experience — handles both "Experience Required: 3+ years" and "3+ years experience" formats
    exp_match = re.search(
        r'Experience\s*(?:Required)?:\s*(\d+(?:\.\d+)?)\+?\s*years?',
        jd_text, re.IGNORECASE
    )
    if not exp_match:
        exp_match = re.search(
            r'(\d+(?:\.\d+)?)\+?\s*years?\s+(?:of\s+)?(?:experience|exp)',
            jd_text, re.IGNORECASE
        )
    min_exp = float(exp_match.group(1)) if exp_match else 0.0

    # Must-have section
    must_section = re.search(
        r'MUST\s*HAVE.*?(?=NICE\s*TO\s*HAVE|RESPONSIBILITIES|COMPENSATION|---|\Z)',
        jd_text, re.DOTALL | re.IGNORECASE
    )
    must_have = []
    if must_section:
        items = re.findall(r'[-•*]\s*(.+)', must_section.group(0))
        must_have = [i.strip() for i in items]

    # Nice-to-have section
    nice_section = re.search(
        r'NICE\s*TO\s*HAVE.*?(?=RESPONSIBILITIES|COMPENSATION|---|\Z)',
        jd_text, re.DOTALL | re.IGNORECASE
    )
    nice_to_have = []
    if nice_section:
        items = re.findall(r'[-•*]\s*(.+)', nice_section.group(0))
        nice_to_have = [i.strip() for i in items]

    return JobDescription(
        jd_id=jd_id,
        title=title,
        company=company,
        must_have=must_have,
        nice_to_have=nice_to_have,
        min_experience=min_exp,
        raw_text=jd_text
    )


# ── Hybrid Scorer ─────────────────────────────────────────────────────────────

class HybridScorer:
    """
    Combines semantic similarity (from RAG) with keyword matching
    for more accurate job-candidate matching.

    Final Score = 0.5 * semantic_score + 0.4 * keyword_score + 0.1 * experience_score
    """

    TECH_KEYWORDS = re.compile(
        r'\b(Python|Java|JavaScript|TypeScript|Go|Rust|C\+\+|Scala|R|SQL|'
        r'React|Node\.js|FastAPI|Django|Flask|Spring|Angular|Vue\.js|Next\.js|'
        r'TensorFlow|PyTorch|scikit-learn|XGBoost|LightGBM|HuggingFace|BERT|LLM|'
        r'AWS|GCP|Azure|Docker|Kubernetes|Terraform|Airflow|Kafka|Spark|'
        r'PostgreSQL|MongoDB|Redis|MySQL|Cassandra|Snowflake|BigQuery|ChromaDB|Pinecone|'
        r'MLflow|Kubeflow|LangChain|LlamaIndex|FAISS|RAG|GPT|'
        r'Prometheus|Grafana|Ansible|Jenkins|ArgoCD|Helm|'
        r'NLP|ML|AI|Deep\s*Learning|Machine\s*Learning)\b',
        re.IGNORECASE
    )

    def extract_keywords(self, text: str) -> List[str]:
        return list(dict.fromkeys(m.group(0) for m in self.TECH_KEYWORDS.finditer(text)))

    def keyword_score(self, jd: JobDescription, metadata: ResumeMetadata) -> Tuple:
        """Score based on keyword overlap between JD and resume."""
        jd_keywords = set(k.lower() for k in self.extract_keywords(jd.raw_text))
        resume_keywords = set(s.lower() for s in metadata.skills)

        # Must-have keywords from JD text
        must_keywords = set(k.lower() for k in self.extract_keywords(' '.join(jd.must_have)))
        nice_keywords = set(k.lower() for k in self.extract_keywords(' '.join(jd.nice_to_have)))

        matched_must = must_keywords & resume_keywords
        matched_nice = nice_keywords & resume_keywords
        matched_skills = list(matched_must | matched_nice)

        # Original skill names (preserve case)
        matched_skill_names = [s for s in metadata.skills if s.lower() in (matched_must | matched_nice)]
        missing_skill_names = [k for k in jd.must_have if
                                not any(k.lower() in s.lower() or s.lower() in k.lower()
                                        for s in metadata.skills)]

        # Score calculation
        must_score = (len(matched_must) / max(len(must_keywords), 1)) * 70
        nice_score = (len(matched_nice) / max(len(nice_keywords), 1)) * 30
        score = must_score + nice_score

        return min(score, 100), matched_skill_names, missing_skill_names[:5]

    def experience_score(self, jd: JobDescription, metadata: ResumeMetadata) -> float:
        """Score based on experience match."""
        exp = metadata.experience_years
        min_exp = jd.min_experience
        if min_exp == 0:
            return 80.0
        if exp >= min_exp:
            return 100.0
        elif exp >= min_exp * 0.7:
            return 70.0
        elif exp > 0:
            return (exp / min_exp) * 60
        return 20.0

    def compute_final_score(
        self,
        semantic_score: float,
        jd: JobDescription,
        metadata: ResumeMetadata
    ) -> Dict:
        """Compute hybrid score combining all signals."""
        kw_score, matched_skills, missing_skills = self.keyword_score(jd, metadata)
        exp_score = self.experience_score(jd, metadata)

        # Weighted combination
        final = (
            0.50 * semantic_score +
            0.40 * kw_score +
            0.10 * exp_score
        )

        return {
            "final_score": round(min(final, 100), 1),
            "semantic_score": round(semantic_score, 1),
            "keyword_score": round(kw_score, 1),
            "experience_score": round(exp_score, 1),
            "matched_skills": matched_skills,
            "missing_skills": missing_skills
        }


# ── Job Matcher ───────────────────────────────────────────────────────────────

# (Tuple already imported at top)

class JobMatcher:
    """
    Main matching engine.
    Uses RAG pipeline for semantic retrieval + HybridScorer for ranking.
    """

    def __init__(self, rag_pipeline: ResumeRAGPipeline):
        self.rag = rag_pipeline
        self.scorer = HybridScorer()

    def _generate_reasoning(
        self,
        metadata: ResumeMetadata,
        jd: JobDescription,
        scores: Dict
    ) -> str:
        """Generate human-readable match reasoning."""
        parts = []

        # Experience
        exp = metadata.experience_years
        min_exp = jd.min_experience
        if exp >= min_exp:
            parts.append(f"✅ {exp:.0f} years experience meets requirement of {min_exp:.0f}+")
        else:
            parts.append(f"⚠️ {exp:.0f} years experience below required {min_exp:.0f}+")

        # Matched skills
        if scores["matched_skills"]:
            top_skills = ", ".join(scores["matched_skills"][:5])
            parts.append(f"✅ Strong skill match: {top_skills}")

        # Missing skills
        if scores["missing_skills"]:
            missing = ", ".join(scores["missing_skills"][:3])
            parts.append(f"⚠️ Missing: {missing}")

        # Current role relevance
        if metadata.current_role and len(metadata.current_role) < 60:
            parts.append(f"📌 Current: {metadata.current_role}")

        # Score breakdown
        parts.append(
            f"📊 Scores — Semantic: {scores['semantic_score']:.0f}, "
            f"Keyword: {scores['keyword_score']:.0f}, "
            f"Experience: {scores['experience_score']:.0f}"
        )

        return " | ".join(parts)

    def _hire_recommendation(self, score: float, missing_skills: List[str]) -> str:
        if score >= 80:
            return "STRONG HIRE — Schedule technical interview immediately"
        elif score >= 65:
            return "HIRE — Recommend for first-round interview"
        elif score >= 50:
            return f"BORDERLINE — Good potential but gaps: {', '.join(missing_skills[:2])}"
        else:
            return "NO HIRE — Significant skill/experience gaps"

    def match(
        self,
        jd_text: str,
        jd_id: str = "JD001",
        top_k: int = 10,
        min_experience: float = None
    ) -> Dict:
        """
        Full matching pipeline:
        1. Parse JD
        2. RAG retrieval (semantic search)
        3. Hybrid scoring for all retrieved candidates
        4. Return ranked JSON output
        """
        start_time = time.time()

        # Parse job description
        jd = parse_job_description(jd_text, jd_id)

        # Determine experience filter
        min_exp = min_experience if min_experience is not None else max(jd.min_experience - 1, 0)

        # RAG retrieval: semantic search
        # For small corpora, retrieve broadly so the hybrid keyword scorer (which is
        # more reliable than the lightweight TF-IDF embedder) gets a fair shot at every
        # candidate rather than being capped by an early, weaker semantic cutoff.
        total_indexed = max(self.rag.vector_store.count(), 1)
        retrieval_k = max(top_k * 3, total_indexed)
        rag_results = self.rag.search(
            query=jd.raw_text,
            top_k=retrieval_k,
            filters={"min_experience": min_exp} if min_exp > 0 else None
        )

        # Get unique candidates from RAG results
        all_metadata = self.rag.get_all_metadata()
        seen_candidates = {}

        for result in rag_results:
            rid = result["resume_id"]
            if rid not in seen_candidates:
                meta = all_metadata.get(rid)
                if meta:
                    seen_candidates[rid] = {
                        "metadata": meta,
                        "semantic_score": result["semantic_score"],
                        "excerpts": [result["content"][:200]]
                    }
                elif result.get("metadata"):
                    # Reconstruct from search result metadata
                    seen_candidates[rid] = {
                        "metadata": result["metadata"],
                        "semantic_score": result["semantic_score"],
                        "excerpts": [result["content"][:200]]
                    }
            else:
                # Accumulate excerpts from multiple sections
                seen_candidates[rid]["excerpts"].append(result["content"][:150])
                # Keep best semantic score
                seen_candidates[rid]["semantic_score"] = max(
                    seen_candidates[rid]["semantic_score"],
                    result["semantic_score"]
                )

        # Score all retrieved candidates
        matches = []
        for rid, data in seen_candidates.items():
            meta = data["metadata"]

            # Convert dict back to ResumeMetadata if needed
            if isinstance(meta, dict):
                meta = ResumeMetadata(**{k: v for k, v in meta.items()
                                         if k in ResumeMetadata.__dataclass_fields__})

            scores = self.scorer.compute_final_score(
                data["semantic_score"], jd, meta
            )

            # Must-have filter: penalize heavily if < 40% must-haves matched
            must_keywords = self.scorer.extract_keywords(' '.join(jd.must_have))
            resume_text_lower = ' '.join(meta.skills).lower()
            must_matched = sum(1 for k in must_keywords if k.lower() in resume_text_lower)
            must_ratio = must_matched / max(len(must_keywords), 1)

            if must_ratio < 0.2:  # Less than 20% must-haves → penalize
                scores["final_score"] *= 0.5

            matches.append(CandidateMatch(
                candidate_name=meta.candidate_name,
                resume_path=meta.file_path,
                match_score=scores["final_score"],
                semantic_score=scores["semantic_score"],
                keyword_score=scores["keyword_score"],
                experience_years=meta.experience_years,
                matched_skills=scores["matched_skills"][:8],
                missing_skills=scores["missing_skills"],
                relevant_excerpts=list(dict.fromkeys(data["excerpts"]))[:3],
                reasoning=self._generate_reasoning(meta, jd, scores),
                hire_recommendation=self._hire_recommendation(
                    scores["final_score"], scores["missing_skills"]
                )
            ))

        # Sort by final score
        matches.sort(key=lambda x: x.match_score, reverse=True)
        top_matches = matches[:top_k]

        latency_ms = round((time.time() - start_time) * 1000, 1)

        # Build output JSON
        output = {
            "job_description": jd.title,
            "company": jd.company,
            "jd_id": jd.jd_id,
            "search_metadata": {
                "total_candidates_evaluated": len(seen_candidates),
                "top_k_returned": len(top_matches),
                "retrieval_latency_ms": latency_ms,
                "jd_required_experience": jd.min_experience,
            },
            "top_matches": [
                {
                    "rank": i + 1,
                    "candidate_name": m.candidate_name,
                    "resume_path": m.resume_path,
                    "match_score": m.match_score,
                    "score_breakdown": {
                        "semantic_score": m.semantic_score,
                        "keyword_score": m.keyword_score,
                        "candidate_experience_years": m.experience_years
                    },
                    "matched_skills": m.matched_skills,
                    "missing_skills": m.missing_skills,
                    "relevant_excerpts": m.relevant_excerpts,
                    "reasoning": m.reasoning,
                    "hire_recommendation": m.hire_recommendation
                }
                for i, m in enumerate(top_matches)
            ]
        }

        return output

    def save_results(self, output: Dict, path: str = "reports/match_results.json"):
        """Save matching results to JSON file."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(output, f, indent=2, ensure_ascii=False)
        return path


import os

# ── Main Demo ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    from resume_rag import ResumeRAGPipeline

    # Build RAG index
    print("=" * 60)
    print("  JOB MATCHER — Initializing RAG Pipeline")
    print("=" * 60)
    pipeline = ResumeRAGPipeline(resume_dir="resumes", persist_dir="vector_store")
    pipeline.build_index()

    matcher = JobMatcher(rag_pipeline=pipeline)

    # Test with all 5 JDs
    jd_texts = []
    jd_file = "job_descriptions/job_descriptions.txt"
    if os.path.exists(jd_file):
        with open(jd_file) as f:
            content = f.read()
        jd_texts = [jd.strip() for jd in content.split('---') if jd.strip()]

    for i, jd_text in enumerate(jd_texts[:5], 1):
        jd_id = f"JD{i:03d}"
        print(f"\n{'='*60}")
        print(f"  Matching JD {jd_id}")
        print(f"{'='*60}")

        result = matcher.match(jd_text, jd_id=jd_id, top_k=5)

        print(f"Position: {result['job_description']} @ {result['company']}")
        print(f"Evaluated: {result['search_metadata']['total_candidates_evaluated']} candidates")
        print(f"Latency: {result['search_metadata']['retrieval_latency_ms']}ms")
        print(f"\nTop Matches:")
        for m in result["top_matches"][:3]:
            print(f"  {m['rank']}. {m['candidate_name']:<20} Score: {m['match_score']:.1f}/100")
            print(f"     {m['hire_recommendation']}")

        # Save to file
        save_path = f"reports/match_{jd_id}.json"
        matcher.save_results(result, save_path)
        print(f"\n  💾 Saved: {save_path}")
