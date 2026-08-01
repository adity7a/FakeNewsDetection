FROM python:3.10-slim

WORKDIR /app

ENV TF_CPP_MIN_LOG_LEVEL=3
ENV PYTHONUNBUFFERED=1

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

RUN python -c "import nltk; nltk.download('stopwords')"

COPY . .

EXPOSE 10000

CMD ["gunicorn", "--bind", "0.0.0.0:10000", "--timeout", "300", "--workers", "1", "--threads", "1", "app:app"]