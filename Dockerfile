# ==== BASE IMAGE ====
FROM python:3.10-slim

# ==== WORKDIR ====
WORKDIR /app

# ==== COPY PROJECT ====
COPY . /app

# ==== INSTALL DEPENDENCIES ====
RUN pip install --no-cache-dir -r requirements.txt

# ==== ENV VARS ====
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app/src

# ==== DEFAULT COMMAND ====
CMD ["python", "predict_batch_advanced.py"]
