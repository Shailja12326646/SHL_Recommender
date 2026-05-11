# SHL Assessment Recommender Agent

An autonomous, conversational agent designed to guide recruiters and hiring managers from vague hiring intents to grounded, actionable assessment shortlists using the SHL catalog. 

## Problem Summary
Finding the right assessment from a large catalog can be overwhelming. Users often lack the exact vocabulary or domain knowledge to search effectively. This system solves that problem by using natural dialogue to clarify requirements, refine choices, and recommend precise assessments.

## Architecture
The system employs a **Deterministic Orchestration** pattern. Rather than relying on a single large LLM prompt to manage state, routing, and search—which often leads to hallucinations and brittle behavior—the architecture decomposes into specialized, deterministic agents:

- **Agent 1 (Retrieval):** A hybrid engine that uses BM25 (word overlap fallback) for exact keyword matches and dense embeddings for semantic search.
- **Agent 2 (Intent):** A conversation state tracker that parses history to extract structured slots (Role, Seniority, Skills).
- **Agent 3 (Workflow):** The orchestrator that routes user inputs to specific logic flows (Recommend, Clarify, Refine, Compare).
- **Agent 10 (Scope Guard):** A hard boundary that strictly refuses prompt injections, legal queries, or off-topic conversation.

*(See `assets/shl_system_architecture.svg` for a visual map)*

## Setup Instructions

1. Clone the repository.
2. Install the requirements:
   ```bash
   pip install -r shl_recommender/requirements.txt
   ```
3. Set up your environment variables:
   Copy `.env.example` to `.env` inside `shl_recommender/` and add your Groq/OpenAI API key.

## Run Instructions

Start the FastAPI service:
```bash
cd shl_recommender
uvicorn main:app --reload
```

## API Usage

### `GET /health`
Returns a readiness status.

### `POST /chat`
Accepts a stateless conversation history array and returns an agent reply and recommendations if ready.
```json
{
  "messages": [
    {"role": "user", "content": "I need a java developer assessment"}
  ]
}
```

## Evaluation
The system is evaluated against 10 public conversational traces using `evaluate.py`.
It strictly measures:
- Schema Compliance
- 100% Catalog adherence (No Hallucinations)
- Mean Recall@10
- Behavior probe pass rates (Scope guarding, refinement handling, etc.)

To run the evaluation harness:
```bash
cd shl_recommender
python evaluate.py --conversations ../dataset/sample_conversations/GenAI_SampleConversations --catalog ../dataset/catalog.txt
```

## Design Decisions
- **Stateless Reconstruction:** Every `/chat` call includes the full history. The system reconstructs the `ConversationState` entirely from this history on every turn. This allows infinite horizontal scaling without sticky sessions or DB dependencies.
- **Catalog-Only Enforcement:** All URLs are injected directly from the deterministic retrieval step, never synthesized by the LLM. 
- **Graceful Degradation:** If the LLM rate-limits or fails, the system falls back to returning a formatted string of the top recommendations, preserving 100% schema compliance and functional UI utility.

## Limitations
- The underlying hybrid search relies on TF-IDF fallback when `rank-bm25` is unavailable, slightly lowering recall on heavily keyword-dependent queries.
- The system focuses exclusively on Individual Test Solutions (not Pre-packaged Solutions) as per assignment scope.
