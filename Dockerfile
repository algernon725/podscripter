FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    ffmpeg \
    git \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
RUN pip install --upgrade pip
RUN pip install git+https://github.com/openai/whisper.git nltk

# Download NLTK data
RUN python -m nltk.downloader punkt

COPY . .

EXPOSE 5000

CMD ["bash"]