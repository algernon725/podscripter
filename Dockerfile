FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    ffmpeg \
    git \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
RUN pip install --upgrade pip
RUN pip install faster-whisper nltk pydub

# Download NLTK data
RUN python -m nltk.downloader punkt

COPY . .

#ENV OMP_NUM_THREADS=8
ENV WHISPER_CACHE_DIR=/app/models

CMD ["bash"]