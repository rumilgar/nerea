FROM python:3.9

WORKDIR /app_code

COPY requirements.txt requirements.txt
RUN ls -la
RUN cat requirements.txt
RUN apt-get update && apt-get install -y portaudio19-dev joe alsa-utils ffmpeg # Añadir la instalación de portaudio19-dev
RUN pip3 install --upgrade pip
RUN pip3 install --no-cache-dir -r requirements.txt

COPY download_whisper_models.py download_whisper_models.py
RUN python3 download_whisper_models.py  # Ejecutar el script para descargar los modelos de Whisper

WORKDIR /app

CMD python3 transcribe_demo.py --model small
