#!/bin/bash

# Docker run command with proper model caching
# This script mounts volumes for Hugging Face (Faster-Whisper, pyannote) and sentence-transformers caches

echo "Running podscripter with model caching..."

docker run -it \
  -v $(pwd)/models/sentence-transformers:/root/.cache/torch/sentence_transformers \
  -v $(pwd)/models/huggingface:/root/.cache/huggingface \
  -v $(pwd)/audio-files:/app/audio-files \
  podscripter

echo "Container finished."
