"""
llm.py — LLM interface (Groq Llama 3 70B / fallback)

Used ONLY for response synthesis and intent clarification as fallback.
All retrieval decisions are deterministic.
"""

from __future__ import annotations

import json
import os
import re
from typing import Any, Dict, List, Optional

import httpx
from dotenv import load_dotenv

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_BASE_URL = "https://api.groq.com/openai/v1"
GROQ_MODEL = "llama-3.3-70b-versatile"

# Fallback to OpenAI-compatible providers
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")


def _call_openai_compat(
    messages: List[Dict[str, str]],
    system: str,
    api_key: str,
    base_url: str,
    model: str,
    max_tokens: int = 800,
    temperature: float = 0.2,
) -> str:
    payload = {
        "model": model,
        "messages": [{"role": "system", "content": system}] + messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
    }
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    response = httpx.post(
        f"{base_url}/chat/completions",
        json=payload,
        headers=headers,
        timeout=5.0,
    )
    response.raise_for_status()
    data = response.json()
    return data["choices"][0]["message"]["content"]


def call_llm(
    messages: List[Dict[str, str]],
    system: str,
    max_tokens: int = 800,
    temperature: float = 0.2,
) -> str:
    """Call LLM with fallback: Groq → OpenAI → simple fallback."""
    if GROQ_API_KEY:
        try:
            return _call_openai_compat(
                messages, system, GROQ_API_KEY, GROQ_BASE_URL, GROQ_MODEL, max_tokens, temperature
            )
        except Exception as e:
            pass  # Suppressed for clean logs

    if OPENAI_API_KEY:
        try:
            return _call_openai_compat(
                messages, system, OPENAI_API_KEY, OPENAI_BASE_URL, OPENAI_MODEL, max_tokens, temperature
            )
        except Exception as e:
            pass  # Suppressed for clean logs

    # Soft fallback
    raise Exception("LLM unavailable — configure GROQ_API_KEY or OPENAI_API_KEY")


# ─────────────────────────────────────────
# System prompts
# ─────────────────────────────────────────

def _load_prompt(name: str) -> str:
    path = os.path.join(os.path.dirname(__file__), "prompts", f"{name}.txt")
    with open(path, "r", encoding="utf-8") as f:
        return f.read()

SYSTEM_RECOMMEND = _load_prompt("recommend")
SYSTEM_CLARIFY = _load_prompt("clarify")
SYSTEM_COMPARE = _load_prompt("compare")
SYSTEM_REFINE = _load_prompt("refine")
SYSTEM_REFUSE = _load_prompt("refusal")


def synthesize_recommendation(
    conversation_history: List[Dict[str, str]],
    retrieved_assessments: List[Any],  # List[SHLAssessment]
    state: Any,  # ConversationState
    is_refinement: bool = False,
) -> str:
    """Generate grounded recommendation explanation."""
    catalog_data = _format_catalog_data(retrieved_assessments)
    system = SYSTEM_REFINE if is_refinement else SYSTEM_RECOMMEND

    context_msg = f"""
CATALOG DATA (use ONLY these assessments):
{catalog_data}

Hiring context:
- Role: {state.role or 'Not specified'}
- Seniority: {state.seniority_raw or state.seniority or 'Not specified'}
- Technical skills needed: {', '.join(state.technical_skills) or 'Not specified'}
- Soft skills: {', '.join(state.soft_skills) or 'Not specified'}
- Personality required: {state.require_personality}
- Cognitive required: {state.require_cognitive}

Write a 2-4 sentence explanation of why these {len(retrieved_assessments)} assessments fit.
{"Note: This is a refinement from the previous recommendations." if is_refinement else ""}
"""
    messages = conversation_history[-6:] + [{"role": "user", "content": context_msg}]
    return call_llm(messages, system, max_tokens=350)


def generate_clarification(
    conversation_history: List[Dict[str, str]],
    missing_slots: List[str],
    state: Any,
) -> str:
    """Generate a multi-slot clarification question."""
    context = f"Missing information needed: {', '.join(missing_slots)}."
    if state.role:
        context += f" Role mentioned: {state.role}."

    messages = conversation_history[-4:] + [{"role": "user", "content": context}]
    return call_llm(messages, SYSTEM_CLARIFY, max_tokens=150)


def generate_comparison(
    conversation_history: List[Dict[str, str]],
    assessment_a: Any,
    assessment_b: Any,
) -> str:
    """Generate grounded comparison between two assessments."""
    catalog_data = f"""
Assessment 1: {assessment_a.name}
URL: {assessment_a.url}
Type: {assessment_a.test_type}
Categories: {', '.join(assessment_a.categories)}
Duration: {assessment_a.duration_minutes or 'Not specified'} minutes
Levels: {', '.join(assessment_a.job_levels[:4])}
Description: {assessment_a.description[:400]}

Assessment 2: {assessment_b.name}
URL: {assessment_b.url}
Type: {assessment_b.test_type}
Categories: {', '.join(assessment_b.categories)}
Duration: {assessment_b.duration_minutes or 'Not specified'} minutes
Levels: {', '.join(assessment_b.job_levels[:4])}
Description: {assessment_b.description[:400]}
"""
    messages = conversation_history[-4:] + [
        {"role": "user", "content": f"Compare these two assessments:\n{catalog_data}"}
    ]
    return call_llm(messages, SYSTEM_COMPARE, max_tokens=400)


def generate_refusal(reason: str = "off_topic") -> str:
    """Return deterministic refusal message."""
    if reason == "injection":
        return "I'm only able to help with SHL assessment selection. Please ask me about assessments for your hiring needs."
    if reason == "legal":
        return "I can't provide legal or compliance advice. I'm here to help you select the right SHL assessments for your hiring needs."
    if reason == "off_topic":
        return "That's outside what I can help with. I'm focused on recommending SHL assessments — feel free to describe the role you're hiring for!"
    return "I can only assist with SHL assessment selection. What role are you hiring for?"


def generate_greeting_response() -> str:
    return ("Hi! I'm here to help you find the right SHL assessments for your hiring needs. "
            "To get started, could you tell me the role you're hiring for and the experience level you're targeting?")


# ─────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────

def _format_catalog_data(assessments: List[Any]) -> str:
    lines = []
    for i, a in enumerate(assessments, 1):
        lines.append(
            f"{i}. {a.name} | Type: {a.test_type} | Duration: {a.duration_minutes or '?'} min | "
            f"Levels: {', '.join(a.job_levels[:3])} | URL: {a.url}\n"
            f"   Description: {a.description[:200]}"
        )
    return "\n".join(lines)
