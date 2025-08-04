# PodScripter

`podscripter` is a lightweight tool designed to transcribe audio using OpenAI's Whisper model inside a Docker container. 

It supports multiple languages with automatic language detection, including English (`en`), Spanish (`es`), French (`fr`), German (`de`), Japanese (`ja`), Russian (`ru`), Czech (`cs`), Italian (`it`), Portuguese (`pt`), Dutch (`nl`), Polish (`pl`), Turkish (`tr`), Arabic (`ar`), Chinese (`zh`), Korean (`ko`), Hindi (`hi`), Swedish (`sv`), Danish (`da`), Norwegian (`no`), and Finnish (`fi`). 

`podscripter` enables users to generate accurate transcriptions locally, making it perfect for platforms like [LingQ](https://www.lingq.com/) where text and audio integration can boost comprehension.

I welcome contributions from people of any skill level to help make this software better! To contribute code, simply clone this repo and submit a pull request. For more information, see the GitHub documentation: [Contributing to a Project](https://docs.github.com/en/get-started/exploring-projects-on-github/contributing-to-a-project).

---

## ✨ Features

- **Local Processing**: No API keys or usage limits, run everything on your own machine.
- **Dockerized Environment**: Easily install and run the tool in an isolated container.
- **Flexible Input**: Supports both audio files (MP3, WAV, etc.) and video files (MP4, etc.).
- **Automatic Language Detection**: Automatically detects the language of your audio content by default.
- **Multi-Language Support**: Supports 20+ languages with manual language selection option.
- **Flexible Output**: Choose your transcription language and output format.
- **Advanced Punctuation Restoration**: Uses advanced NLP techniques to restore proper punctuation in multiple languages.
- **Batch Transcription**: Transcribe multiple files with a simple loop.

---

## 🧰 Requirements

- Apple Mac with an M series processor.
- Windows x86 PC support by modifying the Docker build command

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
>💡 If you’re on an Intel Mac or other architecture, remove --platform linux/arm64

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
| `language`      | (Optional) Language code (`en`, `es`, `fr`, `de`, `ja`, `ru`, `cs`, etc.) - default is auto-detect |
| `output_format` | (Optional) Output format: `txt` or `srt`. - default is `txt`                          |


## 🌍 Supported Languages

PodScripter supports automatic language detection and manual language selection for the following languages:

| Language | Code | Language | Code |
|----------|------|----------|------|
| English | `en` | Japanese | `ja` |
| Spanish | `es` | Russian | `ru` |
| French | `fr` | Czech | `cs` |
| German | `de` | Italian | `it` |
| Portuguese | `pt` | Dutch | `nl` |
| Polish | `pl` | Turkish | `tr` |
| Arabic | `ar` | Chinese | `zh` |
| Korean | `ko` | Hindi | `hi` |
| Swedish | `sv` | Danish | `da` |
| Norwegian | `no` | Finnish | `fi` |

**Note**: Whisper supports many more languages beyond this list. These are the most commonly used ones. When using auto-detection, Whisper will automatically identify the language of your audio content.

## 🔁 Batch Transcription: All Media Files

To transcribe all `.mp3` and `.mp4` files in the audio-files folder with auto-detection (default), run this from inside the container:

  ```bash
  for f in audio-files/*.{mp3,mp4}; do
    python transcribe_sentences.py "$f" audio-files
  done
  ```

## 📚 Why Use This?
When learning a new language, especially through podcasts, having accurate, aligned transcriptions is essential for comprehension and retention. Many language learning apps impose monthly transcription limits or rely on cloud-based AI. This tool gives you full control over your data, with no recurring costs, and the power of Whisper, all on your own hardware.

## 📦 Output
Transcriptions are saved in sentence-separated `.txt` or `.srt`