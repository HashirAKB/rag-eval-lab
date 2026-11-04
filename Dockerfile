FROM python:3.11-slim

WORKDIR /app

# CPU-only torch keeps the image far smaller than the default CUDA build.
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

# Editable install: the package finds data/corpus relative to the source tree.
COPY pyproject.toml README.md ./
COPY src ./src
COPY data ./data
RUN pip install --no-cache-dir -e ".[neural]"

# Download the local models and pre-compute corpus embeddings at build time,
# so the container runs offline and starts in seconds.
RUN python -c "from rag_eval_lab.models import preload; preload()" \
 && python -c "from rag_eval_lab.pipeline import DEFAULT_CONFIG, build_pipeline; build_pipeline(DEFAULT_CONFIG)"

EXPOSE 8000
CMD ["uvicorn", "rag_eval_lab.api:app", "--host", "0.0.0.0", "--port", "8000"]
