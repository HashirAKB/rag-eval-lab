FROM python:3.11-slim

WORKDIR /app

# CPU-only torch keeps the image far smaller than the default CUDA build.
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --no-cache-dir ".[neural]"

COPY data ./data

# Download the local models at build time so the container runs offline.
RUN python -c "from rag_eval_lab.models import preload; preload()"

EXPOSE 8000
CMD ["uvicorn", "rag_eval_lab.api:app", "--host", "0.0.0.0", "--port", "8000"]
