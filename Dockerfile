FROM python:3.9-slim


WORKDIR /app
# Install libraries
COPY requirements.txt /
RUN pip install -r /requirements.txt
RUN apt-get update
RUN apt-get install ffmpeg libsm6 libxext6  -y
# COPY app.py entrypoint.sh ./
COPY . ./

RUN pip install Flask gunicorn  

# launch server with gunicorn
EXPOSE 8080
# CMD ["gunicorn", "--bind=0.0.0.0:8080", "main:app"]
CMD ["gunicorn", "main:app", "--timeout=0", "--preload", \
    "--workers=1", "--threads=4", "--bind=0.0.0.0:8080"]
# CMD [ "python","app.py"]`