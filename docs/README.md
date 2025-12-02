# podscripter

![podscripter](docs/podscripter-logo.jpeg)

## Overview

`podscripter` is a lightweight tool designed to transcribe audio using OpenAI's Whisper model inside a Docker container. It supports multiple languages with automatic language detection, including English (`en`), Spanish (`es`), French (`fr`), and German (`de`). 

`podscripter` enables users to generate accurate transcriptions locally, making it perfect for platforms like [LingQ](https://www.lingq.com/) where text and audio integration can boost comprehension.

---

## Features

- **Local Processing**: No API keys or usage limits, run everything on your own machine.
- **Dockerized Environment**: Easily install and run the tool in an isolated container.
- **Flexible Input**: Supports both audio files (MP3, WAV, etc.) and video files (MP4, etc.).
- **Multiple Output Formats**: Choose between TXT (sentence-separated) or SRT (subtitles).
- **Automatic Language Detection**: Automatically detects the language of your audio content by default.
- **Primary Language Support**: English (en), Spanish (es), French (fr), German (de). Other languages are experimental.
- **Advanced Punctuation Restoration**: Uses Sentence-Transformers for intelligent punctuation restoration, with automatic spaCy-based capitalization.
- **Optional Speaker Diarization**: Detect speaker changes for improved sentence boundaries in multi-speaker content (interviews, conversations, podcasts).
- **Batch Processing**: Transcribe multiple files using simple shell loops.
- **Powered by Whisper**: Uses OpenAI's Whisper model for accurate speech recognition.
- **Hugging Face Integration**: Leverages Hugging Face models and caches for local, offline workflows.

---

## Quickstart

Minimal setup and a single run:

```bash
# Build image
docker build -t podscripter .

# Create cache folders (first time only)
mkdir -p audio-files models/sentence-transformers models/huggingface

# Transcribe one file (TXT output). Replace example.mp3 with your file.
docker run --rm \
  -v $(pwd):/app \
  -v $(pwd)/models/sentence-transformers:/root/.cache/torch/sentence_transformers \
  -v $(pwd)/models/huggingface:/root/.cache/huggingface \
  -v $(pwd)/audio-files:/app/audio-files \
  podscripter python3 /app/podscripter.py \
  /app/audio-files/example.mp3 --output_dir /app/audio-files
```

Notes:
- Compute type defaults to `auto` (no CLI needed).
- VAD is enabled by default. You can disable it with `--no-vad` or adjust padding with `--vad-speech-pad-ms <int>`.

---

## Requirements

- Docker-compatible system (Mac, Linux, Windows with WSL)

---

## Installation

### 1. Install Prerequisites

Make sure you have the following tools installed on your system:

- [Docker](https://www.docker.com) — need help installing? See the beginner guide: [Docker installation for Windows, macOS, and Ubuntu](docs/docker-installation.md)
- [Git](https://git-scm.com/downloads)

### 2. Clone the Repository

Open a terminal and run:
  ```bash
  git clone https://github.com/algernon725/podscripter.git
  cd podscripter
  ```

### 3. Set Up Required Folders

Create folders to store audio files and model data:
  ```bash
  mkdir -p audio-files
  mkdir -p models/sentence-transformers models/huggingface models/pyannote
  ```

This creates the necessary directory structure for caching models:
- `models/huggingface/` - Hugging Face cache (includes Faster-Whisper model repos)
- `models/sentence-transformers/` - Caches sentence embedding models for punctuation restoration
- `models/pyannote/` - Caches speaker diarization models (only needed if using `--enable-diarization`)

### 4. Build the Docker Image

Build the container image that will run the transcription tool:
  ```bash
  docker build -t podscripter .
  ```

### 5. Start the Docker Container

Run the container and mount the folders you just created:
  ```bash
  docker run -it \
    -v $(pwd)/models/sentence-transformers:/root/.cache/torch/sentence_transformers \
    -v $(pwd)/models/huggingface:/root/.cache/huggingface \
    -v $(pwd)/models/pyannote:/root/.cache/pyannote \
    -v $(pwd)/audio-files:/app/audio-files \
    podscripter
  ```
This opens an interactive terminal inside the container. You'll run all transcription commands from here.

**Alternative: Use the caching script**
  ```bash
  ./docker-run-with-cache.sh
  ```

>💡 **Model Caching**: The first run will download models (~1-2 GB). Subsequent runs will use cached models for faster startup.

>⚙️ **NLP Capitalization**: The image enables spaCy-based capitalization by default (NLP_CAPITALIZATION=1). To disable per run, pass `-e NLP_CAPITALIZATION=0` to `docker run`.

## Usage

### Basic Usage

From inside the Docker Container, run:

```bash
python podscripter.py <media_file> --output_dir <output_dir> \
  [--language <code>|auto] [--output_format {txt|srt}] [--single] \
  [--compute-type {auto,int8,int8_float16,int8_float32,float16,float32}] \
  [--beam-size <int>] [--no-vad] [--vad-speech-pad-ms <int>] \
  [--enable-diarization] [--min-speakers <int>] [--max-speakers <int>] \
  [--hf-token <str>] [--quiet|--verbose]
```

**Example:**

To transcribe example.mp3 using default settings (auto-detect language, txt output):

```bash
python podscripter.py audio-files/example.mp3 --output_dir audio-files
```

**Example with video file:**

To transcribe example.mp4:

```bash
python podscripter.py audio-files/example.mp4 --output_dir audio-files
```

## Examples

One example per scenario to keep things concise.

**TXT (default, auto-detect language)**
```bash
python podscripter.py audio-files/example.mp3 --output_dir audio-files
```

**SRT (subtitles)**
```bash
python podscripter.py audio-files/example.mp3 --output_dir audio-files --output_format srt
```

**Single-call (no manual chunking)**
```bash
python podscripter.py audio-files/example.mp3 --output_dir audio-files --single
```
Use `--single` if your hardware can handle longer files in a single call for best context continuity. Default mode uses overlapped chunking with VAD.

**With speaker diarization (interviews/conversations)**

Speaker diarization improves sentence boundaries by detecting when speakers change. First-time setup requires accepting model terms and creating a Hugging Face token:

**Setup (first time only):**
1. **Accept model licenses** (required before token will work):
   - Visit https://huggingface.co/pyannote/speaker-diarization-3.1
   - Click "Agree and access repository"
   - Also accept: https://huggingface.co/pyannote/segmentation-3.0

2. **Create a Hugging Face token**:
   - Go to https://huggingface.co/settings/tokens
   - Click "New token"
   - Choose **"Read"** permission (not "Write")
   - Give it a name (e.g., "podscripter-diarization")
   - Copy the token

3. **Use the token**:
```bash
python podscripter.py audio-files/interview.mp3 --output_dir audio-files \
  --enable-diarization --hf-token YOUR_TOKEN_HERE
```

Or set as environment variable (recommended):
```bash
export HF_TOKEN=YOUR_TOKEN_HERE
python podscripter.py audio-files/interview.mp3 --output_dir audio-files \
  --enable-diarization
```

After first successful download, models are cached in `models/pyannote/` and subsequent runs don't need the token.

### Expected output snippets

English (TXT):
```text
Hello everyone, welcome to our show!
Today, we’re going to talk about travel tips.
```

Spanish (TXT):
```text
Hola a todos, ¡bienvenidos a Españolistos!
Hoy vamos a hablar de algunos consejos de viaje.
```

## Options

| Argument             | Description |
| -------------------- | ----------- |
| `media_file`         | Path to the audio or video file (e.g. `audio-files/example.mp3` or `audio-files/example.mp4`) |
| `--output_dir`       | Directory where the transcription file will be saved |
| `--language`         | Language code. Primary: `en`, `es`, `fr`, `de`. Others are experimental. Default `auto` (auto-detect) |
| `--output_format`    | Output format: `txt` or `srt` (default `txt`) |
| `--single`           | Bypass manual chunking and process the full file in one call |
| `--compute-type`     | Compute type for faster-whisper: `auto`, `int8`, `int8_float16`, `int8_float32`, `float16`, `float32` (default `auto`) |
| `--beam-size`        | Beam size for decoding (default `3`) |
| `--no-vad`           | Disable VAD filtering (default: VAD enabled) |
| `--vad-speech-pad-ms`| Padding in milliseconds when VAD is enabled (default `200`) |
| `--enable-diarization` | Enable speaker diarization for improved sentence boundaries (default: disabled) |
| `--min-speakers`     | Minimum number of speakers (optional, auto-detect if not specified) |
| `--max-speakers`     | Maximum number of speakers (optional, auto-detect if not specified) |
| `--hf-token`         | Hugging Face token for pyannote models (required for first-time download) |
| `--quiet`/`--verbose`| Toggle log verbosity (default `--verbose`) |


## Supported Languages

PodScripter supports automatic language detection and manual language selection for the following languages:

| Language | Code | Language | Code |
|----------|------|----------|------|
| English  | `en` | Spanish  | `es` |
| French   | `fr` | German   | `de` |

**Note**: Whisper can transcribe many additional languages, but only the four listed above have project-level optimization and tests. Other languages are considered experimental.

## Automatic NLP Capitalization (spaCy)

Punctuation restoration uses Sentence-Transformers with automatic spaCy-based capitalization that capitalizes named entities and proper nouns for English, Spanish, French, and German.

- Always enabled - spaCy models are included in the Docker image.
- Automatically capitalizes names, places, and organizations while preserving language-specific connectors like "de", "del", "y", etc.
- For unsupported languages, falls back to the English model.

This feature is CPU-only and uses cached spaCy "sm" models baked into the image.

## Development

See `tests/README.md` for details on running tests.

## Batch Transcription: All Media Files

To transcribe all `.mp3` and `.mp4` files in the audio-files folder with auto-detection (default), run this from inside the container:

  ```bash
  for f in audio-files/*.{mp3,mp4}; do
    python podscripter.py "$f" --output_dir audio-files
  done
  ```

## Why Use This?
When learning a new language, especially through podcasts, having accurate, aligned transcriptions is essential for comprehension and retention. Many language learning apps impose monthly transcription limits or rely on cloud-based AI. This tool gives you full control over your data, with no recurring costs, and the power of Whisper, all on your own hardware.

## Model Caching

Podscripter caches models locally to avoid repeated downloads. Cache locations are created during Installation → "Set Up Required Folders" and are mounted into the container in the run commands above. In short:

- Faster-Whisper (Whisper) models are cached via the Hugging Face Hub under `models/huggingface/` (look for `hub/` entries like `Systran/faster-whisper-*`)
- Sentence-Transformers under `models/sentence-transformers/`
- Pyannote (speaker diarization) under `models/pyannote/` (only needed if using `--enable-diarization`)

Note: The Sentence-Transformers loader first attempts to load from the local cache and prefers offline use when the cache is present (avoids network calls). When caches are warm you may set `HF_HOME` and/or `HF_HUB_OFFLINE=1` to run fully offline.

**To clear cache and re-download models:**
```bash
rm -rf models/sentence-transformers/* models/huggingface/* models/pyannote/*
```

## Output
Transcriptions are saved in sentence-separated `.txt` or `.srt`

---

## Testing

Run the test suite inside Docker with caches mounted. See `tests/README.md` for details.

Quick run (default selection):

```bash
docker run --rm \
  -e NLP_CAPITALIZATION=1 \
  -v $(pwd):/app \
  -v $(pwd)/models/sentence-transformers:/root/.cache/torch/sentence_transformers \
  -v $(pwd)/models/huggingface:/root/.cache/huggingface \
  -v $(pwd)/models/pyannote:/root/.cache/pyannote \
  -v $(pwd)/audio-files:/app/audio-files \
  podscripter python3 /app/tests/run_all_tests.py
```

Optional groups via env flags: `RUN_ALL=1`, `RUN_MULTILINGUAL=1`, `RUN_TRANSCRIPTION=1`, `RUN_DEBUG=1`.

---

## Troubleshooting

- HTTP 429 (rate limiting) during model loads: ensure cache volumes are mounted; the app prefers offline when caches exist. If needed, set `HF_HOME` and consider `HF_HUB_OFFLINE=1` inside the container when caches are warm.
