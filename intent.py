"""
intent.py — Agent 2: Intent + State Extraction

Reconstructs conversation state from full message history (stateless API).
Uses deterministic rules first, LLM as fallback.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Dict, Any

from catalog import resolve_seniority, normalize_role


class Intent(Enum):
    CLARIFY = "clarify"
    RECOMMEND = "recommend"
    REFINE = "refine"
    COMPARE = "compare"
    REFUSE = "refuse"
    GREETING = "greeting"
    UNKNOWN = "unknown"


@dataclass
class ConversationState:
    # Core role info
    role: Optional[str] = None
    seniority: Optional[str] = None  # normalized bucket
    seniority_raw: Optional[str] = None

    # Skills
    technical_skills: List[str] = field(default_factory=list)
    soft_skills: List[str] = field(default_factory=list)

    # Assessment preferences
    require_personality: bool = False
    require_cognitive: bool = False
    require_simulation: bool = False
    require_knowledge: bool = False
    require_development: bool = False

    # Constraints
    max_duration: Optional[int] = None
    remote_required: bool = False
    adaptive_required: bool = False
    language: Optional[str] = None
    category_filter: List[str] = field(default_factory=list)

    # Excluded targets
    excluded_targets: List[str] = field(default_factory=list)

    # Comparison targets
    compare_targets: List[str] = field(default_factory=list)

    # Prior state
    prior_recommendations: List[str] = field(default_factory=list)  # URLs
    has_provided_context: bool = False

    # Job description text (if pasted)
    job_description: Optional[str] = None

    def has_minimum_context(self) -> bool:
        """True if we have enough to make recommendations."""
        return (self.role is not None or self.job_description is not None)

    def missing_required_slots(self) -> List[str]:
        missing = []
        if self.role is None and self.job_description is None:
            missing.append("role")
        return missing

    def to_retrieval_query(self) -> str:
        """Build a rich query string for retrieval."""
        parts = []
        if self.role:
            parts.append(normalize_role(self.role))
        if self.seniority_raw:
            parts.append(self.seniority_raw)
        parts.extend(self.technical_skills)
        parts.extend(self.soft_skills)
        if self.require_personality:
            parts.append("personality behavior OPQ")
        if self.require_cognitive:
            parts.append("cognitive ability aptitude reasoning verify")
        if self.require_simulation:
            parts.append("simulation exercise")
        if self.require_development:
            parts.append("development 360 feedback")
        if self.job_description:
            parts.append(self.job_description[:300])
        return " ".join(parts)


# ─────────────────────────────────────────
# Scope guard — deterministic patterns
# ─────────────────────────────────────────

COMPARE_PATTERNS = [
    r"(difference|differences)\s+between",
    r"compare\s+",
    r"vs\.?\s+",
    r"versus\s+",
    r"which\s+(is|one)\s+(better|best|more suitable)",
    r"how\s+does\s+\S+\s+(compare|differ)",
]

REFINE_PATTERNS = [
    r"(actually|wait|also|and|plus|additionally)\s+(add|include|drop|remove|exclude|change|update|modify)",
    r"(add|include)\s+(personality|cognitive|simulation|knowledge|ability|aptitude|reasoning|development)",
    r"(drop|remove|exclude)\s+",
    r"(too|very)\s+(many|few|long|short|broad|narrow)",
    r"(without|no)\s+(personality|cognitive|simulation)",
    r"(shorter|longer)\s+(assessments?|tests?|durations?)",
    r"(make|keep)\s+it\s+(shorter|longer|simpler|more\s+focused)",
    r"(can\s+you\s+|please\s+)?(adjust|refine|update|change|modify|tweak)\s+(the\s+)?(list|recommendations?|shortlist)",
    r"swap\s+",
]

GREETING_PATTERNS = [
    r"^(hi|hello|hey|howdy|greetings|good\s+(morning|afternoon|evening))[\s!.,]*$",
    r"^(what\s+can\s+you\s+do|help\s+me|how\s+(do|can)\s+you\s+help)[\s?!.]*$",
]

# Seniority extraction
SENIORITY_RE = re.compile(
    r"\b(junior|entry[\s-]?level|intern|graduate|grad|fresh|"
    r"mid[\s-]?level|intermediate|associate|"
    r"senior|sr\.?|lead|principal|staff|"
    r"director|executive|c[\s-]?level|cxo|manager|team\s+lead|"
    r"\d+\s+years?\s+(?:of\s+)?experience)\b",
    re.IGNORECASE
)

DURATION_RE = re.compile(r"\b(\d+)\s*(?:minute|min|minutes)\b", re.IGNORECASE)

LANGUAGE_RE = re.compile(
    r"\b(spanish|french|german|portuguese|dutch|mandarin|chinese|japanese|arabic|"
    r"hindi|italian|russian|korean|swedish|norwegian|danish|finnish|turkish|polish)\b",
    re.IGNORECASE
)

# Common technical skills found in catalog
TECHNICAL_SKILL_PATTERNS = {
    "java": r"\bjava\b(?!\s*script)",
    "python": r"\bpython\b",
    "javascript": r"\b(javascript|js|node\.?js|react|vue)\b",
    "angular": r"\bangular\b",
    "sql": r"\b(sql|mysql|postgresql|oracle|database)\b",
    "aws": r"\b(aws|amazon\s+web\s+services|cloud)\b",
    "spring": r"\bspring\b",
    "docker": r"\b(docker|kubernetes|k8s|containerization)\b",
    "rest": r"\brest\b",
    ".net": r"\b(\.net|dotnet|c#|asp\.net)\b",
    "excel": r"\b(excel|spreadsheet|microsoft\s+excel)\b",
    "word": r"\b(microsoft\s+word|word\s+processing)\b",
    "salesforce": r"\bsalesforce\b",
    "selenium": r"\bselenium\b",
    "machine learning": r"\b(machine\s+learning|ml|deep\s+learning|ai|artificial\s+intelligence)\b",
    "data analysis": r"\b(data\s+anal|tableau|power\s+bi|bi\s+tools)\b",
    "accounting": r"\b(account(ing)?|finance|financial|audit)\b",
    "customer service": r"\b(customer\s+service|customer\s+support|call\s+center|contact\s+center)\b",
    "sales": r"\b(sales|selling|business\s+development|revenue)\b",
    "safety": r"\b(safety|dependability|manufacturing|industrial)\b",
    "leadership": r"\b(leadership|management|managing\s+teams?|people\s+management)\b",
    "communication": r"\b(communication|stakeholder|presentation|writing)\b",
    "healthcare": r"\b(healthcare|medical|hospital|clinical|hipaa)\b",
}

PERSONALITY_TRIGGER = re.compile(
    r"\b(personality|behavior|behaviour|cultural\s+fit|opq|values|motivation|attitude)\b",
    re.IGNORECASE
)
COGNITIVE_TRIGGER = re.compile(
    r"\b(cognitive|reasoning|aptitude|ability|verbal|numerical|inductive|deductive|verify|iq)\b",
    re.IGNORECASE
)
SIMULATION_TRIGGER = re.compile(
    r"\b(simulation|exercise|in[\s-]?tray|inbox|situational|sj[t]?|situational\s+judgment)\b",
    re.IGNORECASE
)
DEVELOPMENT_TRIGGER = re.compile(
    r"\b(development|360|feedback|coaching|learning|upskill|re[\s-]?skill)\b",
    re.IGNORECASE
)
KNOWLEDGE_TRIGGER = re.compile(
    r"\b(knowledge|skills?\s+test|technical\s+test|coding\s+test|proficiency)\b",
    re.IGNORECASE
)


def detect_intent(
    user_text: str,
    state: ConversationState,
    turn_number: int,
) -> Intent:
    """
    Deterministic intent detection with priority ordering.
    1. Refuse (highest priority)
    2. Compare
    3. Refine (if we have prior recommendations)
    4. Clarify / Recommend
    5. Greeting
    """
    text_lower = user_text.lower().strip()

    # 1. Greeting (turn 1 only typically)
    if turn_number <= 2:
        for pattern in GREETING_PATTERNS:
            if re.match(pattern, text_lower, re.IGNORECASE):
                return Intent.GREETING

    # 2. Compare
    for pattern in COMPARE_PATTERNS:
        if re.search(pattern, text_lower, re.IGNORECASE):
            return Intent.COMPARE

    # 3. Refine (only if we have prior recommendations)
    if state.prior_recommendations:
        for pattern in REFINE_PATTERNS:
            if re.search(pattern, text_lower, re.IGNORECASE):
                return Intent.REFINE

    # 6. Check if we have enough info to recommend
    # Update state first, then decide
    return Intent.UNKNOWN  # caller resolves CLARIFY vs RECOMMEND


def extract_state_from_messages(messages: List[Dict[str, str]]) -> ConversationState:
    """
    Reconstruct full ConversationState from message history.
    Processes all user messages in order; later messages override earlier ones.
    """
    state = ConversationState()
    prior_rec_urls = []

    for msg in messages:
        role = msg.get("role", "")
        content = msg.get("content", "")
        content_lower = content.lower()

        if role == "assistant":
            # Extract URLs from prior recommendations
            urls = re.findall(r"https://www\.shl\.com/products/product-catalog/[^\s\)>|]+", content)
            if urls:
                prior_rec_urls = urls
            continue

        if role != "user":
            continue

        # ── Job description detection ──────────────────────────────
        if "job description" in content_lower or "here is" in content_lower:
            if len(content) > 150:
                state.job_description = content[:500]
                state.has_provided_context = True

        # ── Role extraction ────────────────────────────────────────
        role_match = _extract_role(content)
        if role_match:
            state.role = role_match
            state.has_provided_context = True

        # ── Special patterns: graduate scheme, management trainee ──
        if not state.role:
            if re.search(r"\bgraduate\s+(management\s+trainee|scheme|program)\b", content, re.IGNORECASE):
                state.role = "graduate management trainee"
                state.has_provided_context = True
            elif re.search(r"\bcontact\s+cent(?:er|re)\b", content, re.IGNORECASE):
                state.role = "contact centre agent"
                state.has_provided_context = True
            elif re.search(r"\bcall\s+cent(?:er|re)\b", content, re.IGNORECASE):
                state.role = "call centre agent"
                state.has_provided_context = True
            elif re.search(r"\bhealthcare\s+admin\b", content, re.IGNORECASE):
                state.role = "healthcare administrative staff"
                state.has_provided_context = True
            elif re.search(r"\badmin(?:istrative)?\s+assistants?\b", content, re.IGNORECASE):
                state.role = "administrative assistant"
                state.has_provided_context = True
            elif re.search(r"\bsales\s+(rep|representative|team|force)\b", content, re.IGNORECASE):
                state.role = "sales representative"
                state.has_provided_context = True
            elif re.search(r"\bsoftware\s+engineer\b", content, re.IGNORECASE):
                state.role = "software engineer"
                state.has_provided_context = True

        # ── Seniority extraction ───────────────────────────────────
        sen_match = SENIORITY_RE.search(content)
        if sen_match:
            raw = sen_match.group(0)
            bucket = resolve_seniority(raw)
            if bucket:
                state.seniority = bucket
                state.seniority_raw = raw

        # ── Technical skills ───────────────────────────────────────
        for skill, pattern in TECHNICAL_SKILL_PATTERNS.items():
            if re.search(pattern, content, re.IGNORECASE):
                # Check for drop/remove/without
                if re.search(rf"\b(drop|remove|exclude|without|no)\s+{skill}\b", content_lower, re.IGNORECASE):
                    if skill in state.technical_skills:
                        state.technical_skills.remove(skill)
                else:
                    if skill not in state.technical_skills:
                        state.technical_skills.append(skill)

        # ── Assessment type preferences ────────────────────────────
        if PERSONALITY_TRIGGER.search(content):
            state.require_personality = True
        if COGNITIVE_TRIGGER.search(content):
            state.require_cognitive = True
        if SIMULATION_TRIGGER.search(content):
            state.require_simulation = True
        if DEVELOPMENT_TRIGGER.search(content):
            state.require_development = True
        if KNOWLEDGE_TRIGGER.search(content):
            state.require_knowledge = True

        # ── Duration constraint ────────────────────────────────────
        dur_match = DURATION_RE.search(content)
        if dur_match:
            state.max_duration = int(dur_match.group(1))

        # ── Language ──────────────────────────────────────────────
        lang_match = LANGUAGE_RE.search(content)
        if lang_match:
            state.language = lang_match.group(0)

        # ── Remote / Adaptive ─────────────────────────────────────
        if re.search(r"\b(remote|online|virtual)\b", content_lower):
            state.remote_required = True
        if re.search(r"\badaptive\b", content_lower):
            state.adaptive_required = True

        # ── Soft skills ────────────────────────────────────────────
        for skill in ["communication", "leadership", "teamwork", "collaboration",
                       "stakeholder", "presentation", "analytical", "problem solving",
                       "creativity", "interpersonal"]:
            if skill in content_lower:
                if skill not in state.soft_skills:
                    state.soft_skills.append(skill)

        # ── Negation / removal preferences ────────────────────────
        if re.search(r"\b(no|without|drop|remove|exclude)\s+(personality|opq(?:32[rn]?)?)\b", content_lower):
            state.require_personality = False
        if re.search(r"\b(no|without|drop|remove|exclude)\s+(cognitive|ability|aptitude|reasoning|verify(?: g\+)?)\b", content_lower):
            state.require_cognitive = False

        # ── Drop/Remove specific targets ──────────────────────────
        drop_matches = re.findall(
            r"\b(?:drop|remove|exclude|without)\s+(OPQ(?:32[rn]?)?|GSA|Verify[\s\w]*|ADEPT[\s\w]*|DSI|MFS[\s\w]*|SHL[\s\w]+(?:test|assessment))\b",
            content, re.IGNORECASE
        )
        if drop_matches:
            state.excluded_targets = list(set(state.excluded_targets + drop_matches))

        # ── Compare targets ────────────────────────────────────────
        compare_match = re.findall(
            r"\b(OPQ\w*|GSA|Verify[\s\w]*|ADEPT[\s\w]*|DSI|MFS[\s\w]*|SHL[\s\w]+(?:test|assessment))\b",
            content, re.IGNORECASE
        )
        if compare_match:
            state.compare_targets = list(set(state.compare_targets + compare_match))

    state.prior_recommendations = prior_rec_urls
    return state


# ─────────────────────────────────────────
# Role extraction helpers
# ─────────────────────────────────────────

ROLE_RE = re.compile(
    r"(?:hiring\s+(?:a|an|for|\d+)?\s*|looking\s+for\s+(?:a|an|\d+)?\s*|"
    r"for\s+(?:a|an|\d+)?\s+|recruit(?:ing)?\s+(?:a|an|\d+)?\s*|"
    r"position\s+(?:is|for)\s+(?:a|an)?\s*|role\s+(?:is|for)\s+(?:a|an)?\s*|"
    r"we\s+need\s+(?:a|an|\d+)?\s+|need\s+(?:a|an|\d+)?\s+|"
    r"screening\s+\d+\s+"
    r")"
    r"((?:senior|junior|mid[\s-]?level|entry[\s-]?level|lead|principal|staff|"
    r"[\w]+\s+)?[\w]+(?:\s+[\w]+){0,3}?(?:\s+(?:developer|engineer|analyst|manager|"
    r"specialist|consultant|representative|officer|associate|architect|designer|"
    r"administrator|coordinator|director|lead|executive|scientist|researcher|"
    r"agent|agents|staff|worker|workers|trainee|trainees|intern|operator|technician))\w*)",
    re.IGNORECASE
)

BARE_ROLE_RE = re.compile(
    r"\b((?:senior|junior|mid[\s-]?level|entry[\s-]?level|lead|principal|staff|"
    r"[\w]+\s+)?[\w]+(?:\s+[\w]+){0,2}\s+"
    r"(?:developer|engineer|analyst|manager|specialist|consultant|representative|"
    r"officer|associate|architect|designer|administrator|coordinator|director|"
    r"lead|executive|scientist|researcher|trainee|intern|agent|operator|technician|assistant)s?)\b",
    re.IGNORECASE
)

def _extract_role(text: str) -> Optional[str]:
    """Extract role name from user message."""
    # Try explicit hiring patterns first
    m = ROLE_RE.search(text)
    if m:
        return m.group(1).strip()
    # Fallback: find bare role title
    m = BARE_ROLE_RE.search(text)
    if m:
        return m.group(1).strip()
    # Last resort: check for single known role keywords
    for kw in ["developer", "engineer", "analyst", "manager", "sales", "accountant",
                "nurse", "doctor", "teacher", "driver", "operator", "technician"]:
        if kw in text.lower():
            # Find the broader phrase around the keyword
            pattern = rf"[\w\s-]{{0,20}}{kw}[\w\s-]{{0,10}}"
            m2 = re.search(pattern, text, re.IGNORECASE)
            if m2:
                return m2.group(0).strip()[:60]
    return None
