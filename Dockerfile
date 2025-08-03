FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1

WORKDIR /app

RUN apt-get update && apt-get install -y \
    build-essential \
    libssl-dev \
    libffi-dev \
    git \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt requirements.txt
RUN python -m venv /opt/venv \
    && /opt/venv/bin/pip install --no-cache-dir --upgrade pip \
    && /opt/venv/bin/pip install --no-cache-dir -r requirements.txt

ENV PATH="/opt/venv/bin:$PATH"

RUN mkdir -p /app/data /app/models

COPY data/ /app/data/
COPY models/ /app/models/

COPY . .

ENV PYTHONPATH="/app"

EXPOSE 5000

CMD ["gunicorn", "-b", "0.0.0.0:5000", "interfaces.web_app.app:app"]

