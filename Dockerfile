FROM python:3.11-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy all source files
COPY . .

# Precompute embeddings and cache
ENV HF_HOME=/app/.cache
RUN python precompute.py

EXPOSE 8000

ENV CATALOG_PATH=catalog.txt
ENV EMBED_CACHE_PATH=cache/embeddings.pkl

CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT:-8000}"]
