"""
workflow.py — Agents 3-10: Workflow Router + All Specialized Agents

Deterministic orchestration. Routes to the right flow based on intent + state.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Tuple

from catalog import SHLAssessment
from intent import (
    ConversationState, Intent,
    detect_intent, extract_state_from_messages,
)
from llm import (
    synthesize_recommendation, generate_clarification,
    generate_comparison, generate_refusal, generate_greeting_response,
)
from retrieval import RetrievalEngine


# ─────────────────────────────────────────
# Response schema
# ─────────────────────────────────────────

def make_response(
    reply: str,
    recommendations: Optional[List[Dict]] = None,
    end_of_conversation: bool = False,
) -> Dict:
    return {
        "reply": reply,
        "recommendations": recommendations or [],
        "end_of_conversation": end_of_conversation,
    }


def assessment_to_rec(a: SHLAssessment) -> Dict:
    return {
        "name": a.name,
        "url": a.url,
        "test_type": a.test_type,
    }


# ─────────────────────────────────────────
# Scope / safety guard
# ─────────────────────────────────────────

INJECTION_PATTERNS = [
    r"ignore\s+(previous|prior|above|all)\s+instructions",
    r"forget\s+(everything|all|previous)",
    r"you\s+are\s+now",
    r"jailbreak",
    r"bypass\s+.{0,20}(rules|guidelines)",
    r"new\s+persona",
    r"act\s+as\s+(if|a)\s+",
    r"disregard\s+",
]

LEGAL_PATTERNS = [
    r"legal\s+advice",
    r"hiring\s+policy\s+(legal|illegal|complian)",
    r"(age|race|gender|religion)\s+discriminat",
    r"salary\s+(recommend|range|advice)",
    r"what\s+should\s+i\s+pay",
    r"is\s+(this|it)\s+(legal|illegal|lawful|compliant)",
]

OFF_TOPIC_HARD = [
    r"\b(recipe|cooking|weather|sport[^s]|movie|music|song|joke|dating|relationship|"
    r"bitcoin|crypto|invest(ment)?|politic|religion)\b",
]


def check_scope(text: str) -> Optional[str]:
    """Return refusal reason string if out of scope, else None."""
    text_lower = text.lower()
    for p in INJECTION_PATTERNS:
        if re.search(p, text_lower):
            return "injection"
    for p in LEGAL_PATTERNS:
        if re.search(p, text_lower):
            return "legal"
    for p in OFF_TOPIC_HARD:
        if re.search(p, text_lower, re.IGNORECASE):
            # only refuse if no hiring/assessment context
            hiring_kws = ["assess", "shl", "test", "hire", "hiring", "role", "candidate", "recruit"]
            if not any(kw in text_lower for kw in hiring_kws):
                return "off_topic"
    return None


# ─────────────────────────────────────────
# Clarification agent
# ─────────────────────────────────────────

CLARIFICATION_TEMPLATES = {
    ("role", "seniority"): (
        "I'd be happy to help! Could you tell me: (1) what role you're hiring for, "
        "and (2) the experience level or seniority you're targeting?"
    ),
    ("role",): (
        "What role or position are you hiring for?"
    ),
    ("seniority",): (
        "What's the seniority or experience level for this role? "
        "(e.g. entry-level, mid-level, senior, manager, executive)"
    ),
}


def clarification_agent(
    messages: List[Dict],
    state: ConversationState,
    missing: List[str],
) -> Dict:
    """Ask for missing required slots — multi-slot when possible."""
    key = tuple(sorted(missing))
    template = CLARIFICATION_TEMPLATES.get(key)

    if template:
        reply = template
    else:
        # Use LLM for more nuanced clarifications
        try:
            reply = generate_clarification(messages, missing, state)
        except Exception:
            reply = (
                f"To recommend the right assessments, I need a bit more information. "
                f"Could you tell me: {' and '.join(missing)}?"
            )

    return make_response(reply, recommendations=[], end_of_conversation=False)


# ─────────────────────────────────────────
# Recommendation agent
# ─────────────────────────────────────────

def recommendation_agent(
    messages: List[Dict],
    state: ConversationState,
    engine: RetrievalEngine,
    is_refinement: bool = False,
) -> Dict:
    """Retrieve + rank + synthesize recommendation."""

    query = state.to_retrieval_query()

    # Determine seniority buckets to filter on
    seniority_buckets = [state.seniority] if state.seniority else None

    # Determine excluded ids
    excluded_ids = []
    if getattr(state, "excluded_targets", None):
        for name in state.excluded_targets:
            a = engine.get_by_name(name)
            if a:
                excluded_ids.append(a.id)

    # Retrieve candidates
    results = engine.retrieve(
        query=query,
        top_k=10,
        seniority_buckets=seniority_buckets,
        require_personality=False,  # we handle this in post-processing
        max_duration=state.max_duration,
        remote_only=state.remote_required,
        adaptive_only=state.adaptive_required,
        language=state.language,
        excluded_ids=excluded_ids,
    )

    if not results:
        # Retry without seniority filter
        results = engine.retrieve(
            query=query,
            top_k=10,
            max_duration=state.max_duration,
        )

    assessments = [a for a, _ in results]

    # ── Post-processing: inject required types ─────────────────────
    assessments = _inject_required_types(assessments, state, engine, query)

    # Cap at 10
    assessments = assessments[:10]

    if not assessments:
        return make_response(
            "I wasn't able to find assessments matching all your criteria. "
            "Could you relax some constraints — for example, remove the duration limit or broaden the skill set?",
            recommendations=[],
            end_of_conversation=False,
        )

    # ── Generate explanation ───────────────────────────────────────
    try:
        reply = synthesize_recommendation(messages, assessments, state, is_refinement=is_refinement)
    except Exception as e:
        print(f"[LLM] synthesis failed: {e}")
        role_str = state.role or "the role"
        reply = (
            f"Here {'are' if len(assessments) > 1 else 'is'} "
            f"{len(assessments)} assessment{'s' if len(assessments) > 1 else ''} "
            f"for {role_str} ({state.seniority_raw or state.seniority or 'the level specified'}).\n\n"
            + "\n".join(a.url for a in assessments)
        )

    recs = [assessment_to_rec(a) for a in assessments]
    return make_response(reply, recommendations=recs, end_of_conversation=False)


def _inject_required_types(
    assessments: List[SHLAssessment],
    state: ConversationState,
    engine: RetrievalEngine,
    query: str,
) -> List[SHLAssessment]:
    """Ensure required assessment types are included."""
    existing_ids = {a.id for a in assessments}
    extra = []

    if state.require_personality:
        if not any("Personality & Behavior" in a.categories for a in assessments):
            pers_results = engine.retrieve(
                query=query + " occupational personality questionnaire opq32r",
                top_k=3,
                seniority_buckets=[state.seniority] if state.seniority else None,
            )
            for a, _ in pers_results:
                if a.id not in existing_ids and "Personality & Behavior" in a.categories:
                    extra.append(a)
                    existing_ids.add(a.id)
                    break

    if state.require_cognitive:
        if not any("Ability & Aptitude" in a.categories for a in assessments):
            cog_results = engine.retrieve(
                query=query + " verify interactive g+",
                top_k=3,
            )
            for a, _ in cog_results:
                if a.id not in existing_ids and "Ability & Aptitude" in a.categories:
                    extra.append(a)
                    existing_ids.add(a.id)
                    break

    if state.require_simulation:
        if not any("Simulations" in a.categories for a in assessments):
            sim_results = engine.retrieve(
                query=query + " simulation exercise in-tray",
                top_k=3,
            )
            for a, _ in sim_results:
                if a.id not in existing_ids and "Simulations" in a.categories:
                    extra.append(a)
                    existing_ids.add(a.id)
                    break

    if state.require_development:
        if not any("Development & 360" in a.categories for a in assessments):
            dev_results = engine.retrieve(
                query=query + " development 360 feedback",
                top_k=3,
            )
            for a, _ in dev_results:
                if a.id not in existing_ids and "Development & 360" in a.categories:
                    extra.append(a)
                    existing_ids.add(a.id)
                    break

    return assessments + extra


# ─────────────────────────────────────────
# Refinement agent
# ─────────────────────────────────────────

def refinement_agent(
    messages: List[Dict],
    state: ConversationState,
    engine: RetrievalEngine,
) -> Dict:
    """State-aware re-ranking. Preserves context, applies new constraints."""
    return recommendation_agent(messages, state, engine, is_refinement=True)


# ─────────────────────────────────────────
# Comparison agent
# ─────────────────────────────────────────

COMPARE_NAME_RE = re.compile(
    r"\b(OPQ32[rn]?|OPQ\s+\w+|GSA|Global\s+Skills\s+Assessment|"
    r"Verify\s*[A-Z\+\s]*|ADEPT[\s\-\w]*|DSI|Dependability[\s\w]+|"
    r"MFS[\s\w]+|Verify\s*G\+|Automata[\s\w]*|"
    r"[A-Z][a-zA-Z0-9\s\.\-]{2,40}(?:test|assessment|questionnaire|simulation|report|inventory))\b",
    re.IGNORECASE
)


def comparison_agent(
    messages: List[Dict],
    state: ConversationState,
    engine: RetrievalEngine,
    last_user_message: str,
) -> Dict:
    """Grounded comparison between two named assessments."""

    # Extract assessment names from the message + state
    names_found = COMPARE_NAME_RE.findall(last_user_message)
    names_found = list(dict.fromkeys(n.strip() for n in names_found))  # dedupe, preserve order

    if len(names_found) < 2:
        # Try state compare_targets
        names_found = list(dict.fromkeys(names_found + state.compare_targets))

    if len(names_found) < 2:
        return make_response(
            "Could you name the two assessments you'd like me to compare? "
            "For example: 'Compare OPQ32r and GSA'.",
            recommendations=[],
        )

    a1 = engine.get_by_name(names_found[0])
    a2 = engine.get_by_name(names_found[1])

    missing = []
    if not a1:
        missing.append(names_found[0])
    if not a2:
        missing.append(names_found[1])

    if missing:
        return make_response(
            f"I couldn't find '{' and '.join(missing)}' in the SHL catalog. "
            "Please check the assessment name and try again.",
            recommendations=[],
        )

    try:
        reply = generate_comparison(messages, a1, a2)
    except Exception as e:
        print(f"[LLM] comparison failed: {e}")
        reply = (
            f"**{a1.name}**: {a1.description[:200]}\n\n"
            f"**{a2.name}**: {a2.description[:200]}\n\n"
            "The key differences are in their purpose and target traits as described above."
        )

    recs = [assessment_to_rec(a1), assessment_to_rec(a2)]
    return make_response(reply, recommendations=recs)


# ─────────────────────────────────────────
# Output validator
# ─────────────────────────────────────────

def validate_response(response: Dict, engine: RetrievalEngine) -> Dict:
    """
    Validate schema and hallucination guard.
    - Ensures all URLs exist in catalog
    - Ensures 1-10 recommendations (or 0 if not committed)
    - Ensures required fields present
    """
    # Schema compliance
    if "reply" not in response:
        response["reply"] = ""
    if "recommendations" not in response:
        response["recommendations"] = []
    if "end_of_conversation" not in response:
        response["end_of_conversation"] = False

    # URL hallucination guard
    clean_recs = []
    for rec in response.get("recommendations", []):
        url = rec.get("url", "")
        if engine.is_valid_url(url):
            clean_recs.append(rec)
        else:
            print(f"[Validator] Dropped invalid URL: {url}")

    # Deduplicate by URL
    seen_urls = set()
    deduped = []
    for rec in clean_recs:
        if rec["url"] not in seen_urls:
            deduped.append(rec)
            seen_urls.add(rec["url"])

    # Enforce 1-10 if recommendations present
    if len(deduped) > 10:
        deduped = deduped[:10]

    response["recommendations"] = deduped

    # If we have recs, ensure name and test_type are present
    for rec in response["recommendations"]:
        if "name" not in rec:
            rec["name"] = "Unknown"
        if "test_type" not in rec:
            rec["test_type"] = "K"

    return response


# ─────────────────────────────────────────
# Main Orchestrator
# ─────────────────────────────────────────

def orchestrate(
    messages: List[Dict[str, str]],
    engine: RetrievalEngine,
) -> Dict:
    """
    Full pipeline:
    1. Scope guard
    2. State extraction
    3. Intent detection
    4. Route to appropriate agent
    5. Validate response
    """

    if not messages:
        return make_response(
            "Hello! I'm here to help you select SHL assessments. "
            "What role are you hiring for?"
        )

    last_user_msg = ""
    for msg in reversed(messages):
        if msg.get("role") == "user":
            last_user_msg = msg.get("content", "")
            break

    # ── Step 1: Scope guard ────────────────────────────────────────
    refusal_reason = check_scope(last_user_msg)
    if refusal_reason:
        return make_response(generate_refusal(refusal_reason))

    # ── Step 2: State extraction ───────────────────────────────────
    state = extract_state_from_messages(messages)
    turn_number = sum(1 for m in messages if m.get("role") == "user")

    # ── Step 3: Intent detection ───────────────────────────────────
    intent = detect_intent(last_user_msg, state, turn_number)

    # ── Step 4: Route ──────────────────────────────────────────────
    if intent == Intent.GREETING:
        response = make_response(generate_greeting_response())

    elif intent == Intent.COMPARE:
        response = comparison_agent(messages, state, engine, last_user_msg)

    elif intent == Intent.REFINE:
        response = refinement_agent(messages, state, engine)

    else:
        # RECOMMEND or UNKNOWN → check if we have enough context
        missing = state.missing_required_slots()

        if missing:
            # Check if this is turn 1+ with minimal info but vague — clarify
            response = clarification_agent(messages, state, missing)
        elif state.has_minimum_context():
            # Have enough → recommend
            is_refine = (intent == Intent.REFINE) or bool(state.prior_recommendations)
            response = recommendation_agent(messages, state, engine, is_refinement=is_refine and bool(state.prior_recommendations))
        else:
            response = clarification_agent(messages, state, missing or ["role", "seniority"])

    # ── Step 5: Validate ───────────────────────────────────────────
    response = validate_response(response, engine)

    # ── Turn cap guard: if at turn 8, end conversation ─────────────
    if turn_number >= 7 and not response["recommendations"]:
        # Force a recommendation attempt on last allowed turn
        if state.has_minimum_context():
            response = recommendation_agent(messages, state, engine)
            response = validate_response(response, engine)

    return response
