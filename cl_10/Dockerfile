FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# System deps (lightweight; safe for pillow/tf)
RUN apt-get update \
    && apt-get install -y --no-install-recommends libglib2.0-0 libgl1 \
    && rm -rf /var/lib/apt/lists/*

COPY animal_app/requirements.txt /app/animal_app/requirements.txt
RUN pip install --no-cache-dir -r /app/animal_app/requirements.txt

COPY animal_app /app/animal_app
COPY animal-10.keras /app/animal-10.keras

EXPOSE 7860
CMD ["uvicorn", "animal_app.app:app", "--host", "0.0.0.0", "--port", "7860"]
