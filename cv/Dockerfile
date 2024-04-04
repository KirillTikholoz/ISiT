FROM python:3.10

WORKDIR /cv

ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

RUN apt-get update && apt-get install -y libgl1-mesa-glx

RUN pip install --upgrade pip
ADD requirements.txt requirements.txt
RUN pip install --no-cache-dir --no-cache -r requirements.txt

COPY app /cv/app
COPY dataRepository /cv/dataRepository
COPY images /cv/images

CMD ["uvicorn", "app.api:app", "--host", "0.0.0.0", "--port", "8080"]