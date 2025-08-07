FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    ffmpeg \
    git \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
RUN pip install --upgrade pip
RUN pip install faster-whisper pydub
RUN pip install sentence-transformers

# No longer need NLTK - using simple sentence splitting

COPY . .

#ENV OMP_NUM_THREADS=8
ENV WHISPER_CACHE_DIR=/app/models
ENV HF_HOME=/root/.cache/huggingface

CMD ["bash"]