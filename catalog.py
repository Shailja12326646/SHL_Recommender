"""
catalog.py — Agent 0: Catalog Processing Agent

Loads, normalizes, and prepares SHL catalog for retrieval.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

# ─────────────────────────────────────────
# Normalization tables
# ─────────────────────────────────────────

SENIORITY_MAP = {
    # raw job_level → normalized bucket
    "entry-level": "entry",
    "graduate": "entry",
    "general population": "entry",
    "professional individual contributor": "mid",
    "mid-professional": "mid",
    "supervisor": "mid",
    "front line manager": "manager",
    "manager": "manager",
    "director": "senior",
    "executive": "executive",
}

# User-stated seniority aliases → catalog bucket
SENIORITY_ALIASES = {
    "junior": "entry",
    "entry": "entry",
    "entry level": "entry",
    "entry-level": "entry",
    "intern": "entry",
    "fresh": "entry",
    "graduate": "entry",
    "grad": "entry",
    "mid": "mid",
    "mid level": "mid",
    "mid-level": "mid",
    "intermediate": "mid",
    "associate": "mid",
    "4 years": "mid",
    "5 years": "mid",
    "3 years": "mid",
    "senior": "senior",
    "sr": "senior",
    "sr.": "senior",
    "lead": "senior",
    "principal": "senior",
    "staff": "senior",
    "director": "senior",
    "manager": "manager",
    "mgr": "manager",
    "team lead": "manager",
    "vp": "executive",
    "executive": "executive",
    "c-level": "executive",
    "cxo": "executive",
    "ceo": "executive",
    "cto": "executive",
    "cfo": "executive",
}

# category key → test_type letter (for schema compliance)
CATEGORY_TO_TYPE = {
    "Knowledge & Skills": "K",
    "Ability & Aptitude": "A",
    "Personality & Behavior": "P",
    "Competencies": "C",
    "Simulations": "S",
    "Biodata & Situational Judgment": "B",
    "Development & 360": "D",
    "Assessment Exercises": "E",
}

ROLE_ALIASES = {
    "java dev": "java developer",
    "java engineer": "java developer",
    "backend engineer": "software engineer backend",
    "frontend engineer": "software engineer frontend",
    "fe engineer": "frontend developer",
    "swe": "software engineer",
    "ml engineer": "machine learning engineer",
    "ai engineer": "machine learning engineer",
    "data scientist": "data science analyst",
    "devops engineer": "cloud infrastructure engineer",
    "qa engineer": "quality assurance engineer",
    "test engineer": "quality assurance engineer",
    "sales rep": "sales representative",
    "bdr": "business development representative",
    "sdr": "sales development representative",
    "hr manager": "human resources manager",
    "hr business partner": "human resources business partner",
    "customer success": "customer service",
    "cs rep": "customer service representative",
    "admin": "administrative assistant",
}

# skill keywords that map well to catalog terms
SKILL_KEYWORD_MAP = {
    "java": ["java", "spring", "hibernate", "maven"],
    "python": ["python", "django", "flask", "pandas"],
    "sql": ["sql", "database", "mysql", "postgresql"],
    "javascript": ["javascript", "js", "node", "react", "angular", "vue"],
    "aws": ["aws", "cloud", "amazon web services"],
    "devops": ["docker", "kubernetes", "ci/cd", "jenkins"],
    "communication": ["business communication", "verbal reasoning", "writing"],
    "leadership": ["leadership", "management", "opq", "personality"],
    "personality": ["opq", "personality", "behavior", "behaviour"],
    "cognitive": ["verify", "deductive", "inductive", "numerical reasoning", "verbal reasoning"],
    "sales": ["sales", "selling", "commercial"],
    "customer service": ["customer service", "contact center", "call center"],
    "safety": ["safety", "dependability", "dsi"],
    "finance": ["accounts", "financial", "excel", "accounting"],
    "excel": ["microsoft excel", "excel 365", "spreadsheet"],
    "word": ["microsoft word", "word 365"],
}


@dataclass
class SHLAssessment:
    id: str
    name: str
    url: str
    description: str
    categories: List[str]
    job_levels: List[str]  # raw
    job_level_buckets: List[str]  # normalized
    duration_minutes: Optional[int]
    languages: List[str]
    remote_support: bool
    adaptive_support: bool
    test_type: str  # comma-joined type letters
    searchable_text: str  # for BM25
    embedding_text: str  # for dense retrieval


def _parse_duration(raw: str) -> Optional[int]:
    """Extract integer minutes from duration string."""
    if not raw:
        return None
    m = re.search(r"(\d+)", raw)
    return int(m.group(1)) if m else None


def _derive_test_type(categories: List[str]) -> str:
    letters = []
    seen = set()
    for cat in categories:
        letter = CATEGORY_TO_TYPE.get(cat, "")
        if letter and letter not in seen:
            letters.append(letter)
            seen.add(letter)
    return ",".join(letters) if letters else "K"


def _build_searchable_text(item: dict) -> str:
    """Rich BM25 searchable string with synonyms."""
    parts = [item.get("name", "")]

    desc = item.get("description", "")
    parts.append(desc)

    # Add category terms
    for cat in item.get("keys", []):
        parts.append(cat)

    # Add level terms
    for lv in item.get("job_levels", []):
        parts.append(lv)

    # Add languages
    langs = item.get("languages", [])
    if langs:
        parts.append(" ".join(langs[:3]))

    # Inject skill synonyms based on name/description
    combined = (item.get("name", "") + " " + desc).lower()
    for skill, synonyms in SKILL_KEYWORD_MAP.items():
        if any(s in combined for s in synonyms):
            parts.extend(synonyms)

    return " ".join(p for p in parts if p)


def _build_embedding_text(item: dict) -> str:
    """Concise semantic text for embedding."""
    name = item.get("name", "")
    desc = item.get("description", "")[:300]
    cats = ", ".join(item.get("keys", []))
    levels = ", ".join(item.get("job_levels", []))
    return f"{name}. {cats}. {levels}. {desc}"


def _normalize_level_buckets(levels: List[str]) -> List[str]:
    buckets = set()
    for lv in levels:
        bucket = SENIORITY_MAP.get(lv.lower(), "")
        if bucket:
            buckets.add(bucket)
    return list(buckets)


def _fix_broken_strings(raw_text: str) -> str:
    """
    Fix catalog JSON where some string values span multiple lines.
    Strategy: if a line has an odd number of (unescaped) quotes, the string
    is unclosed — merge with the next line until it closes.
    """
    lines = raw_text.splitlines(keepends=True)
    result = []
    i = 0
    while i < len(lines):
        line = lines[i]
        stripped = line.rstrip("\n\r")

        # Only try to fix lines that contain a string value opening
        if ('": "' in stripped or '":"' in stripped):
            quote_count = stripped.count('"') - stripped.count('\\"')
            if quote_count % 2 == 1:  # unclosed string
                while i + 1 < len(lines):
                    i += 1
                    next_line = lines[i].rstrip("\n\r")
                    stripped = stripped + " " + next_line.strip()
                    qc = stripped.count('"') - stripped.count('\\"')
                    if qc % 2 == 0:
                        break
                result.append(stripped + "\n")
                i += 1
                continue

        result.append(line)
        i += 1
    return "".join(result)


def load_catalog(path: str) -> List[SHLAssessment]:
    """Load and normalize the SHL catalog from JSON file."""
    raw_text = Path(path).read_bytes().decode("utf-8", errors="replace")
    fixed = _fix_broken_strings(raw_text)
    raw_data = json.loads(fixed)

    assessments = []
    for item in raw_data:
        if item.get("status") != "ok":
            continue

        cats = item.get("keys", [])
        levels_raw = item.get("job_levels", [])
        level_buckets = _normalize_level_buckets(levels_raw)

        assessment = SHLAssessment(
            id=item.get("entity_id", ""),
            name=item.get("name", ""),
            url=item.get("link", ""),
            description=item.get("description", ""),
            categories=cats,
            job_levels=levels_raw,
            job_level_buckets=level_buckets,
            duration_minutes=_parse_duration(item.get("duration_raw", "")),
            languages=item.get("languages", []),
            remote_support=item.get("remote", "no").lower() == "yes",
            adaptive_support=item.get("adaptive", "no").lower() == "yes",
            test_type=_derive_test_type(cats),
            searchable_text=_build_searchable_text(item),
            embedding_text=_build_embedding_text(item),
        )
        assessments.append(assessment)

    return assessments


# ─────────────────────────────────────────
# Helper: resolve user seniority string
# ─────────────────────────────────────────

def resolve_seniority(text: str) -> Optional[str]:
    """Map user-stated seniority text to a normalized bucket."""
    text_lower = text.lower().strip()
    # direct match
    if text_lower in SENIORITY_ALIASES:
        return SENIORITY_ALIASES[text_lower]
    # substring match
    for alias, bucket in sorted(SENIORITY_ALIASES.items(), key=lambda x: -len(x[0])):
        if alias in text_lower:
            return bucket
    return None


def normalize_role(text: str) -> str:
    """Normalize role aliases."""
    text_lower = text.lower().strip()
    for alias, normalized in ROLE_ALIASES.items():
        if alias in text_lower:
            return text_lower.replace(alias, normalized)
    return text_lower
