# PodScripter

**Transcribe your podcasts. No limits. No cloud.**

---

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
  or
- Windows x86 PC support by modifying the Docker build command (but I need testers to validate this)

---

## 🚀 Setup

1. Install [Docker](https://www.docker.com) and [Git](https://git-scm.com/downloads)

2. Clone the GitHub repo:
  ```bash
  git clone https://github.com/algernon725/podscripter.git
  ```

3. Create the Docker volume `audio-files` and `models` folders to store your Docker-related files and audio inputs/outputs:
  ```bash
  mkdir -p podscripter/audio-files
  mkdir -p podscripter/models
  cd podscripter
  ```

4. Build the Docker image:
  ```bash
  docker build --platform linux/arm64 -t podscripter .
  ```

5. Run the Docker container:
  ```bash
  docker run --platform linux/arm64 -it \
  -v $(pwd)/models:/root/.cache/whisper \
  -v $(pwd)/audio-files:/app/audio-files \
  podscripter
  ```

## 📄 Command-line Usage
*Please note that these commands need to be run from the command prompt inside of your running Docker container, which will appear after you run the Docker container in step 5 above.*

Usage: python transcribe_sentences.py <audio_file> <output_dir> [language (default 'en')] [model_size (default 'medium')] [output_format (txt|srt, default 'txt')]"

To transcribe an audio file named `example.mp3` from the command prompt inside the container:
  ```bash
  python transcribe_sentences.py audio-files/example.mp3 audio-files
  ```

For example, to transcribe an audio file named `example.mp3` containing Spanish speech, you can specify the language `es`:

  ```bash
  python transcribe_sentences.py audio-files/example.mp3 audio-files es
  ```

To transcribe an audio file named `example.mp3` containing French speech, with the `medium` model size, and output in `.srt` format, use:

  ```bash
  python transcribe_sentences.py audio-files/example.mp3 audio-files fr medium srt
  ```

## Command-Line Arguments

| Argument     | Description                                                                 |
| ------------ | --------------------------------------------------------------------------- |
| `audio_file` | Path to the audio file you want to transcribe                               |
| `output_dir` | Directory where the transcription will be saved                             |
| `language`   | (Optional) Output language code (`en`, `es`, `fr`, `de`) Default: `en` for English                 |
| `model_size` | (Optional) Whisper model size to use (`small.multilingual`, `medium`, `large-v2`). Larger models are more accurate, but require more RAM and are slower. Default: `medium` requires ~5GB RAM|
| `output_format` | (Optional) Output format to use (`txt`, `srt`). Default: `txt`


## Batch Transcription (all `.mp3` files):
To transcribe all mp3 files from the command prompt in spanish inside the container:
  ```bash
  for f in audio-files/*.mp3; do
    python transcribe_sentences.py "$f" audio-files es
  done
  ```

## 📚 Why Use This?
When learning a new language, especially through podcasts, having accurate, aligned transcriptions is essential for comprehension and retention. Many language learning apps impose monthly transcription limits or rely on cloud-based AI. This tool gives you full control over your data, with no recurring costs, and the power of Whisper, all on your own hardware.

## 📦 Output
Transcriptions are saved in sentence-separated `.txt` or `.srt` format, ready for import into language learning platforms. The tool automatically restores proper punctuation using DeepSegment for better sentence segmentation in supported languages (English, Spanish, German, French).

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

Copyright (c) 2025 Algernon Greenidge