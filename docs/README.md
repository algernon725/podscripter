# PodScripter

`podscripter` is a lightweight tool designed to transcribe audio using OpenAI's Whisper model inside a Docker container. 

It supports multiple languages with automatic language detection, including English (`en`), Spanish (`es`), French (`fr`), and German (`de`). 

`podscripter` enables users to generate accurate transcriptions locally, making it perfect for platforms like [LingQ](https://www.lingq.com/) where text and audio integration can boost comprehension.

---

## ✨ Features

- **Local Processing**: No API keys or usage limits, run everything on your own machine.
- **Dockerized Environment**: Easily install and run the tool in an isolated container.
- **Flexible Input**: Supports both audio files (MP3, WAV, etc.) and video files (MP4, etc.).
- **Multiple Output Formats**: Choose between TXT (sentence-separated) or SRT (subtitles).
- **Automatic Language Detection**: Automatically detects the language of your audio content by default.
- **Primary Language Support**: English (en), Spanish (es), French (fr), German (de). Other languages are experimental.
- **Advanced Punctuation Restoration**: Uses Sentence-Transformers for intelligent punctuation restoration, with an optional spaCy-based capitalization pass.
- **Batch Processing**: Transcribe multiple files using simple shell loops.
- **Powered by Whisper**: Uses OpenAI's Whisper model for accurate speech recognition.
- **HuggingFace Integration**: Leverages HuggingFace models and caches for local, offline workflows.

---

## 🧰 Requirements

- Apple Mac with an M series processor.
- Other architectures supported by modifying the Docker build command

---

## 🚀 Quick Setup Guide

### 1. Install Prerequisites

Make sure you have the following tools installed on your system:

- [Docker](https://www.docker.com)
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
  mkdir -p models/whisper models/sentence-transformers models/huggingface
  ```

This creates the necessary directory structure for caching models:
- `models/whisper/` - Caches Whisper speech recognition models
- `models/sentence-transformers/` - Caches sentence embedding models for punctuation restoration
- `models/huggingface/` - Caches HuggingFace transformer models and datasets

### 4. Build the Docker Image

Build the container image that will run the transcription tool:
  ```bash
  docker build --platform linux/arm64 -t podscripter .
  ```
>💡 If you’re on an Intel Mac or other architecture, remove --platform linux/arm64

### 5. Start the Docker Container

Run the container and mount the folders you just created:
  ```bash
  docker run --platform linux/arm64 -it \
  -v $(pwd)/models/whisper:/app/models \
  -v $(pwd)/models/sentence-transformers:/root/.cache/torch/sentence_transformers \
  -v $(pwd)/models/huggingface:/root/.cache/huggingface \
  -v $(pwd)/audio-files:/app/audio-files \
  podscripter
  ```
This opens an interactive terminal inside the container. You'll run all transcription commands from here.
>💡 If you’re on an Intel Mac or other architecture, remove --platform linux/arm64

**Alternative: Use the caching script**
  ```bash
  ./docker-run-with-cache.sh
  ```

>💡 **Model Caching**: The first run will download models (~1-2 GB). Subsequent runs will use cached models for faster startup.

>⚙️ **NLP Capitalization**: The image enables spaCy-based capitalization by default (NLP_CAPITALIZATION=1). To disable per run, pass `-e NLP_CAPITALIZATION=0` to `docker run`.

## 📄 How to Use The Transcription Tool

### Basic Usage

From inside the Docker Container, run:

```bash
python transcribe_sentences.py <media_file> <output_dir> [language] [output_format]
```

**Example:**

To transcribe example.mp3 using default settings (auto-detect language, txt output):

```bash
python transcribe_sentences.py audio-files/example.mp3 audio-files
```

**Example with video file:**

To transcribe example.mp4:

```bash
python transcribe_sentences.py audio-files/example.mp4 audio-files
```

## Optional Parameters

You can optionally customize the transcription language, and output format.

**Example: Spanish Transcription**

```bash
python transcribe_sentences.py audio-files/example.mp3 audio-files es
```

**Example: French with .srt output**

```bash
python transcribe_sentences.py audio-files/example.mp3 audio-files fr srt
```

**Example: Force auto-detection**

```bash
python transcribe_sentences.py audio-files/example.mp3 audio-files auto
```

## Command-Line Options

| Argument        | Description                                                                           |
| --------------- | ------------------------------------------------------------------------------------- |
| `media_file`    | Path to the audio or video file (e.g. audio-files/example.mp3 or audio-files/example.mp4) |
| `output_dir`    | Directory where the transcription file will be saved                                  |
| `language`      | (Optional) Language code. Primary: `en`, `es`, `fr`, `de`. Others are experimental. Default is auto-detect. |
| `output_format` | (Optional) Output format: `txt` or `srt`. - default is `txt`                          |


## 🌍 Supported Languages

PodScripter supports automatic language detection and manual language selection for the following languages:

| Language | Code | Language | Code |
|----------|------|----------|------|
| English  | `en` | Spanish  | `es` |
| French   | `fr` | German   | `de` |

**Note**: Whisper can transcribe many additional languages, but only the four listed above have project-level optimization and tests. Other languages are considered experimental.

## 🅰️ Optional NLP Capitalization (spaCy)

Punctuation restoration uses Sentence-Transformers. You can optionally enable an NLP capitalization pass (spaCy) that capitalizes named entities and proper nouns for English, Spanish, French, and German.

- Enabled by default inside the container (Dockerfile sets `NLP_CAPITALIZATION=1`).
- Disable per run:
  ```bash
  docker run ... -e NLP_CAPITALIZATION=0 ...
  ```
- Re-enable per run (if disabled in your custom image):
  ```bash
  docker run ... -e NLP_CAPITALIZATION=1 ...
  ```

This pass is CPU-only and cached via spaCy “sm” models baked into the image.

## 🔁 Batch Transcription: All Media Files

To transcribe all `.mp3` and `.mp4` files in the audio-files folder with auto-detection (default), run this from inside the container:

  ```bash
  for f in audio-files/*.{mp3,mp4}; do
    python transcribe_sentences.py "$f" audio-files
  done
  ```

## 📚 Why Use This?
When learning a new language, especially through podcasts, having accurate, aligned transcriptions is essential for comprehension and retention. Many language learning apps impose monthly transcription limits or rely on cloud-based AI. This tool gives you full control over your data, with no recurring costs, and the power of Whisper, all on your own hardware.

## 🗄️ Model Caching

PodScripter uses several AI models that are cached locally to avoid re-downloading:

- **Whisper Models** (`models/whisper/`): Speech recognition models (~1-2 GB)
- **Sentence-Transformers** (`models/sentence-transformers/`): Punctuation restoration models (~100-200 MB)
- **HuggingFace Cache** (`models/huggingface/`): Transformer models and datasets (~50-100 MB)

**To clear cache and re-download models:**
```bash
rm -rf models/whisper/* models/sentence-transformers/* models/huggingface/*
```

## 📦 Output
Transcriptions are saved in sentence-separated `.txt` or `.srt`