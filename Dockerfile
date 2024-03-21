FROM python:3.10

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

ADD requirements.txt requirements.txt
RUN pip install --no-cache-dir --upgrade -r requirements.txt

COPY scraper /app/scraper

CMD ["uvicorn", "scraper.api:app", "--host", "0.0.0.0", "--port", "8000"]