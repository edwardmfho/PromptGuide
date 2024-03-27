FROM python:3.11-slim

WORKDIR /app

RUN apt-get -y update  && apt-get install -y \
  apt-utils \
  build-essential \
&& rm -rf /var/lib/apt/lists/*

RUN pip install --upgrade pip setuptools \
    && pip install cython==0.29.35

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

CMD gunicorn -w 3 -k uvicorn.workers.UvicornWorker app:app --bind 0.0.0.0:$PORT
