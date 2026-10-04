FROM python:3.11-slim

WORKDIR /app

# Install dependencies first so Docker can cache this layer
COPY pyproject.toml .

RUN mkdir -p app && \
    touch app/__init__.py && \
    pip install --no-cache-dir .

# Copy application code after dependencies are installed
COPY app ./app

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]