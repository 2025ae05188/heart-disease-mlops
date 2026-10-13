# Use a Python runtime compatible with our pinned dependencies.
FROM python:3.12-slim

# Configure Python and the application package path.
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app/src

WORKDIR /app

# Install only the inference dependencies.
COPY requirements-serving.txt requirements-monitoring.txt ./
RUN python -m pip install --no-cache-dir -r requirements-monitoring.txt

# Copy the application source code.
COPY src/ ./src/

# Copy the existing Section 4 champion model.
COPY models/champion/model.pkl ./models/champion/model.pkl

# FastAPI/Uvicorn service port.
EXPOSE 8000

# Start the API server.
CMD ["uvicorn", "heart_disease_ml.serving.api:app", "--host", "0.0.0.0", "--port", "8000"]