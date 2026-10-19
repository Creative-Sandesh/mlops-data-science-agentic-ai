FROM python:3.12-slim

WORKDIR /app
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# Reproducible install from the committed lock file (W17: replaces `pip install -r requirements.txt`)
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project

COPY app/ ./app/
COPY agent/ ./agent/
COPY src/ ./src/
COPY prompts/ ./prompts/
COPY configs/ ./configs/
COPY data/ ./data/

EXPOSE 8000
CMD ["uv", "run", "--no-sync", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
