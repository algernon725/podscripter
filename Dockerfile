FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libsndfile1 \
    git \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Install Python dependencies in fewer layers for better caching
# pyannote.audio 4.x requires torch>=2.8, torchaudio>=2.8, torchcodec>=0.7
# torchcodec 0.8 is incompatible with torch 2.8 (ABI mismatch), pin to 0.7.0
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir \
    faster-whisper \
    pydub \
    torch==2.8.0 \
    torchaudio==2.8.0 \
    torchcodec==0.7.0 \
    soundfile \
    sentence-transformers==5.2.2 \
    spacy==3.8.11 \
    pyannote.audio==4.0.4

# Install spaCy language models (3.8.0 for spacy 3.8.x)
RUN pip install --no-cache-dir \
    https://github.com/explosion/spacy-models/releases/download/es_core_news_sm-3.8.0/es_core_news_sm-3.8.0-py3-none-any.whl \
    https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.8.0/en_core_web_sm-3.8.0-py3-none-any.whl \
    https://github.com/explosion/spacy-models/releases/download/fr_core_news_sm-3.8.0/fr_core_news_sm-3.8.0-py3-none-any.whl \
    https://github.com/explosion/spacy-models/releases/download/de_core_news_sm-3.8.0/de_core_news_sm-3.8.0-py3-none-any.whl

# Copy application code (done last for optimal layer caching)
COPY . .

ENV HF_HOME=/root/.cache/huggingface
# pyannote.audio 4.x uses HF_HOME for model caching (PYANNOTE_CACHE no longer used)

CMD ["bash"]