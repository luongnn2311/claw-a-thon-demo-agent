FROM python:3.10-slim

WORKDIR /app

# ── System deps ───────────────────────────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# ── Python deps ───────────────────────────────────────────────────────────────
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt \
    && pip install --no-cache-dir huggingface_hub

# ── Download embedding model into the image at build time ─────────────────────
# This avoids any runtime download and works fully offline in the container.
RUN python - <<'EOF'
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id="sentence-transformers/all-MiniLM-L6-v2",
    local_dir="/app/models/all-MiniLM-L6-v2",
)
print("Embedding model downloaded.")
EOF

# ── Copy source code ──────────────────────────────────────────────────────────
COPY fraud_analytics/ ./fraud_analytics/
COPY data/knowledge/  ./data/knowledge/
COPY main.py .
COPY server.py .
COPY frontend/ ./frontend/

# ── Runtime ───────────────────────────────────────────────────────────────────
# Note: vector store is built at container startup (FAISS uses AVX2 which
# can't run under QEMU during cross-platform build on Apple Silicon).
ENV PYTHONUNBUFFERED=1
EXPOSE 8080

CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8080"]
