FROM python:3.10-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy data, app and src
COPY data/ ./data/
COPY src/ ./src/
COPY app/ ./app/

EXPOSE 7860

ENV MODEL_ID=""
ENV PYTHONPATH=/app/src

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "7860"]
