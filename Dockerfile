FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    ffmpeg \
    git \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Install Python dependencies in fewer layers for better caching
# Pin torch/torchaudio versions for pyannote.audio compatibility (torchaudio <2.8 required)
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir \
    faster-whisper \
    pydub \
    torch==2.2.0 \
    torchaudio==2.2.0 \
    sentence-transformers \
    spacy==3.7.4 \
    pyannote.audio==3.3.2

# Install spaCy language models
RUN pip install --no-cache-dir \
    https://github.com/explosion/spacy-models/releases/download/es_core_news_sm-3.7.0/es_core_news_sm-3.7.0-py3-none-any.whl \
    https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.7.0/en_core_web_sm-3.7.0-py3-none-any.whl \
    https://github.com/explosion/spacy-models/releases/download/fr_core_news_sm-3.7.0/fr_core_news_sm-3.7.0-py3-none-any.whl \
    https://github.com/explosion/spacy-models/releases/download/de_core_news_sm-3.7.0/de_core_news_sm-3.7.0-py3-none-any.whl

# Copy application code (done last for optimal layer caching)
COPY . .

ENV HF_HOME=/root/.cache/huggingface

CMD ["bash"]