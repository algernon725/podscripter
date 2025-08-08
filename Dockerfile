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
RUN pip install spacy==3.7.4
RUN python -m spacy download es_core_news_sm
RUN python -m spacy download en_core_web_sm
RUN python -m spacy download fr_core_news_sm
RUN python -m spacy download de_core_news_sm

# No longer need NLTK - using simple sentence splitting

COPY . .

#ENV OMP_NUM_THREADS=8
ENV WHISPER_CACHE_DIR=/app/models
ENV HF_HOME=/root/.cache/huggingface
ENV NLP_CAPITALIZATION=1

CMD ["bash"]