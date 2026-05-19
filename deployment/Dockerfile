FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

COPY requirements.txt pyproject.toml README.md ./
COPY src ./src
COPY dashboard ./dashboard
COPY app.py ./app.py
COPY config ./config
COPY data ./data
COPY models ./models

RUN mkdir -p images \
    && pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt \
    && pip install --no-cache-dir -e .

EXPOSE 8501

CMD ["streamlit", "run", "app.py", "--server.address=0.0.0.0", "--server.port=8501"]

