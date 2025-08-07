#!/bin/bash

# Docker run command with proper model caching
# This script mounts volumes for both Whisper and sentence-transformers caches

echo "Running podscripter with model caching..."

docker run --platform linux/arm64 -it \
  -v $(pwd)/models/whisper:/app/models \
  -v $(pwd)/models/sentence-transformers:/root/.cache/torch/sentence_transformers \
  -v $(pwd)/models/huggingface:/root/.cache/huggingface \
  -v $(pwd)/audio-files:/app/audio-files \
  podscripter-debug

echo "Container finished."
