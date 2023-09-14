FROM python:3.10-slim
ENV PORT=9090
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . ./
CMD exec gunicorn  --bind :$PORT --workers 1 --threads 2 --timeout 0 app:app
RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y
