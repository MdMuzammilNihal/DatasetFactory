FROM python:3.13-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/hf-cache

# The pipeline reads config/ and models/ relative to the working directory.
WORKDIR /app
COPY pyproject.toml ./
COPY dataset_factory ./dataset_factory
COPY config ./config
COPY models ./models
RUN pip install -e ".[api]"

RUN useradd --create-home app \
    && mkdir -p /app/runs /hf-cache \
    && chown -R app:app /app/runs /hf-cache
USER app

# Job results and the downloaded tokenizer survive container restarts.
VOLUME ["/app/runs", "/hf-cache"]
EXPOSE 8000

CMD ["python", "-m", "dataset_factory", "serve", "--host", "0.0.0.0", "--port", "8000", "--out", "/app/runs"]
