#!/bin/bash

# Docker run command with proper model caching
# This script mounts the Hugging Face cache (Faster-Whisper, pyannote models)

echo "Running podscripter with model caching..."

docker run -it \
  -v $(pwd)/models/huggingface:/root/.cache/huggingface \
  -v $(pwd)/audio-files:/app/audio-files \
  podscripter

echo "Container finished."
