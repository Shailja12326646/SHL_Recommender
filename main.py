"""
main.py — FastAPI service for SHL Assessment Recommender

Endpoints:
  GET  /health  → {"status": "ok"}
  POST /chat    → {"reply": str, "recommendations": [...], "end_of_conversation": bool}
"""

from __future__ import annotations

import os
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, validator

# ─────────────────────────────────────────
# Global state (initialized at startup)
# ─────────────────────────────────────────
_engine = None
_startup_time = None

CATALOG_PATH = os.getenv("CATALOG_PATH", "catalog.txt")


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _engine, _startup_time
    t0 = time.time()
    print("[Startup] Loading catalog...")

    from catalog import load_catalog
    from retrieval import RetrievalEngine

    # Find catalog file
    catalog_path = CATALOG_PATH
    if not Path(catalog_path).exists():
        # Try relative paths
        for p in ["catalog.txt", "../catalog.txt", "/app/catalog.txt", "/mnt/user-data/uploads/catalog.txt"]:
            if Path(p).exists():
                catalog_path = p
                break

    assessments = load_catalog(catalog_path)
    print(f"[Startup] Loaded {len(assessments)} assessments")

    _engine = RetrievalEngine(assessments)
    _startup_time = time.time() - t0
    print(f"[Startup] Ready in {_startup_time:.2f}s")

    yield

    print("[Shutdown] Bye!")


app = FastAPI(
    title="SHL Assessment Recommender",
    version="1.0.0",
    lifespan=lifespan,
)


# ─────────────────────────────────────────
# Request / Response models
# ─────────────────────────────────────────

class Message(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    messages: List[Message] = Field(..., min_items=1)

    @validator("messages")
    def validate_roles(cls, v):
        for msg in v:
            if msg.role not in ("user", "assistant", "system"):
                raise ValueError(f"Invalid role: {msg.role}")
        return v


class Recommendation(BaseModel):
    name: str
    url: str
    test_type: str


class ChatResponse(BaseModel):
    reply: str
    recommendations: List[Recommendation]
    end_of_conversation: bool


# ─────────────────────────────────────────
# Endpoints
# ─────────────────────────────────────────

@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    if _engine is None:
        raise HTTPException(status_code=503, detail="Service not ready")

    messages = [{"role": m.role, "content": m.content} for m in request.messages]

    from workflow import orchestrate
    result = orchestrate(messages, _engine)

    return ChatResponse(
        reply=result.get("reply", ""),
        recommendations=[
            Recommendation(
                name=r["name"],
                url=r["url"],
                test_type=r.get("test_type", "K"),
            )
            for r in result.get("recommendations", [])
        ],
        end_of_conversation=result.get("end_of_conversation", False),
    )


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    print(f"[Error] {exc}")
    return JSONResponse(
        status_code=500,
        content={
            "reply": "An internal error occurred. Please try again.",
            "recommendations": [],
            "end_of_conversation": False,
        },
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False)
