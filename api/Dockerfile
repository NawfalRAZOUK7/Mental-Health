# Mental Health Viz — prediction API (FastAPI + scikit-learn).
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    MHP_VERSION=v1

WORKDIR /app

# Install API dependencies first (better layer caching).
COPY api/requirements.txt api/requirements.txt
RUN pip install --upgrade pip && pip install -r api/requirements.txt

# Copy the project (needs src/ + v1/data_clean + data_raw for the model).
COPY . .

EXPOSE 8000
HEALTHCHECK CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://localhost:8000/health').status==200 else 1)"

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
