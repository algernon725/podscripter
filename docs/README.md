# PodScripter

`podscripter` is a lightweight tool designed to transcribe audio using OpenAI’s Whisper model inside a Docker container. It supports popular languages including English (`en`), Spanish (`es`), French (`fr`), and German (`de`). Originally, I created this project to help with my own language learning journey. My goal was to build a free podcast transcription tool, practice coding in Python, and learn how to use Docker. `podscripter` enables users to generate accurate transcriptions locally, making it perfect for platforms like [LingQ](https://www.lingq.com/) where text and audio integration can boost comprehension.

I welcome contributions from people of any skill level to help make this software better! Right now, the most urgent need is for someone with x86 Windows hardware to help build and test the tool on Windows, as well as for testers who can try out French and German transcriptions. To contribute code, simply clone this repo and submit a pull request. For more information, see the GitHub documentation: [Contributing to a Project](https://docs.github.com/en/get-started/exploring-projects-on-github/contributing-to-a-project).

---

## ✨ Features

- **Local Processing**: No API keys or usage limits, run everything on your own machine.
- **Dockerized Environment**: Easily install and run the tool in an isolated container.
- **Flexible Output**: Choose your transcription language and model size.
- **Punctuation Restoration**: Uses advanced NLP techniques to restore proper punctuation in English, Spanish, German, and French transcriptions.
- **Batch Transcription**: Transcribe multiple files with a simple loop.

---

## 🧰 Requirements

- Apple Mac with an M series processor.
- Windows x86 PC support by modifying the Docker build command (testers needed)

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
  mkdir -p models
  ```

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
  -v $(pwd)/models:/root/.cache/whisper \
  -v $(pwd)/audio-files:/app/audio-files \
  podscripter
  ```
  This opens an interactive terminal inside the container. You'll run all transcription commands from here.

## 📄 How to Use The Transcription Tool

### Basic Usage

From inside the Docker Container, run:

```bash
python transcribe_sentences.py <audio_file> <output_dir> [language] [model_size] [output_format]
```

**Example:**

To transcribe example.mp3 using default settings (english language, medium model, txt output):

```bash
python transcribe_sentences.py audio-files/example.mp3 audio-files
```

## Optional Parameters

You can optionally customize the transcription language, model size, and output format.

**Example: Spanish Transcription**

```bash
python transcribe_sentences.py audio-files/example.mp3 audio-files es
```

**Example: French with a specific model and .srt output**

```bash
python transcribe_sentences.py audio-files/example.mp3 audio-files fr medium srt
```

## Command-Line Options

| Argument        | Description                                                                           |
| --------------- | ------------------------------------------------------------------------------------- |
| `audio_file`    | Path to the audio file (e.g. audio-files/example.mp3)                                 |
| `output_dir`    | Directory where the transcription file will be saved                                  |
| `language`      | (Optional) Language code (`en`, `es`, `fr`, `de`) - default is `en`                   |
| `model_size`    | (Optional) Whisper model  to use (`small`, `medium`, `large-v2`) - default is `medium`|
| `output_format` | (Optional) Output format: `txt` or `srt`. - default is `txt`                          |

>🔧 Tip: Larger models (like large-v2) produce more accurate transcriptions but use more memory and are slower.


## 🔁 Batch Transcription: All MP3 Files

To transcribe all `.mp3` files in the audio-files folder (e.g., in Spanish), run this from inside the container:

  ```bash
  for f in audio-files/*.mp3; do
    python transcribe_sentences.py "$f" audio-files es
  done
  ```

## 📚 Why Use This?
When learning a new language, especially through podcasts, having accurate, aligned transcriptions is essential for comprehension and retention. Many language learning apps impose monthly transcription limits or rely on cloud-based AI. This tool gives you full control over your data, with no recurring costs, and the power of Whisper, all on your own hardware.

## 📦 Output
Transcriptions are saved in sentence-separated `.txt` or `.srt` format, ready for import into language learning platforms.

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

Copyright (c) 2025 Algernon Greenidge