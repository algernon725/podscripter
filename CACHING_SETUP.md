# Docker Model Caching Setup

This guide explains how to set up Docker volume caching to avoid re-downloading models every time you run the container.

## Problem

When running the podscripter container, large model files are downloaded every time:
- **Whisper models**: `model.bin`, `model.safetensors` (for speech recognition)
- **Sentence-transformers models**: `paraphrase-multilingual-MiniLM-L12-v2` (for punctuation restoration)

## Solution

### 1. Directory Structure

Create the following directory structure in your project:

```
podscripter/
├── models/
│   ├── sentence-transformers/ # Sentence-transformers cache
│   └── huggingface/          # HuggingFace cache (transformers, datasets, Faster-Whisper)
├── audio-files/              # Your audio files
└── docker-run-with-cache.sh  # Caching script
```

### 2. Create Directories

```bash
mkdir -p models/sentence-transformers models/huggingface
```

### 3. Docker Run Command

Use this command instead of your current one:

```bash
docker run -it \
  -v $(pwd)/models/sentence-transformers:/root/.cache/torch/sentence_transformers \
  -v $(pwd)/models/huggingface:/root/.cache/huggingface \
  -v $(pwd)/audio-files:/app/audio-files \
  podscripter
```

### 4. Or Use the Script

```bash
./docker-run-with-cache.sh
```

## What Gets Cached

### Faster-Whisper Models (via Hugging Face Hub)
- Speech recognition models under Hugging Face cache
- Downloaded by `faster-whisper` from repositories like `Systran/faster-whisper-*`

### Sentence-Transformers (`/root/.cache/torch/sentence_transformers`)
- Embedding models for punctuation restoration
- Model: `paraphrase-multilingual-MiniLM-L12-v2`
- Downloaded by `sentence-transformers` library

### HuggingFace Cache (`/root/.cache/huggingface`)
- Transformer models, datasets, and all HuggingFace resources
- Used by sentence-transformers and other HuggingFace libraries
- Includes tokenizers, configs, model weights, and datasets
- Note: Uses `HF_HOME` environment variable (deprecated `TRANSFORMERS_CACHE` removed)

## Verification

After the first run, check that files are being cached:

```bash
# Check sentence-transformers cache
ls -la models/sentence-transformers/

# Check HuggingFace cache
ls -la models/huggingface/
```

## Benefits

- **Faster startup**: No model downloads on subsequent runs
- **Offline capability**: Works without internet after first run
- **Bandwidth savings**: Models downloaded only once
- **Consistent performance**: Same models used every time

## Troubleshooting

### If models still download every time:

1. **Check volume mounts**: Ensure the directories exist and are writable
2. **Check permissions**: Make sure the container can write to the mounted volumes
3. **Check environment variables**: Verify the cache directories are set correctly
4. **Check for deprecation warnings**: Ensure you're using the latest Dockerfile with `HF_HOME` instead of deprecated `TRANSFORMERS_CACHE`

### To force re-download:

```bash
# Remove cached models
rm -rf models/sentence-transformers/* models/huggingface/*
```

## Expected File Sizes

- **Whisper models**: ~1-2 GB (depending on model size)
- **Sentence-transformers**: ~100-200 MB
- **HuggingFace cache**: ~50-100 MB

## Rebuilding the Container

If you update the Dockerfile, rebuild with:

```bash
docker build -t podscripter .
```

The cached models will persist across container rebuilds.
