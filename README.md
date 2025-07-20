# whisper-docker-nltk

**Local audio transcription for language learners using OpenAI Whisper, powered by Docker.**

---

`whisper-docker-nltk` is a lightweight tool designed to transcribe Spanish-language audio (or other supported languages) using OpenAI’s Whisper model inside a Docker container. Originally created for language learners, it enables users to generate accurate transcriptions locally—perfect for platforms like [LingQ](https://www.lingq.com/) where text + audio integration improves comprehension.

---

## ✨ Features

- **Local Processing**: No API keys or usage limits—run everything on your own machine.
- **Dockerized Environment**: Easily install and run the tool in an isolated container.
- **Flexible Output**: Choose your transcription language and model size.
- **Batch Transcription**: Transcribe multiple files with a simple loop.

---

## 🧰 Requirements

- Apple Mac with M1, M2, M3, or M4 processor.

---

## 🚀 Getting Started

1. Install [Docker](https://www.docker.com) and [Git](https://git-scm.com/downloads)

2. Create the project folders to store your Docker-related files and audio inputs/outputs:
```bash
mkdir -p whisper-docker-nltk/audio-files
mkdir -p whisper-docker-nltk/models
cd whisper-docker-nltk
```

3. Build the Docker image:
```bash
docker build --platform linux/arm64 -t whisper-ai-nltk .
```

4. Run the container:
```bash
docker run --platform linux/arm64 -it \
  -v $(pwd)/models:/root/.cache/whisper \
  -v $(pwd)/audio-files:/app/audio-files \
  whisper-ai-nltk
```

## 📄 Usage
To transcribe an audio file from the command prompt inside the container:
```bash
python transcribe_sentences.py audio-files/example.mp3 audio-files
```

## Command-Line Arguments

| Argument     | Description                                                                 |
| ------------ | --------------------------------------------------------------------------- |
| `audio_file` | Path to the audio file you want to transcribe                               |
| `output_dir` | Directory where the transcription will be saved                             |
| `language`   | (Optional) Output language code for example `en` for English (default: `es` for Spanish)                 |
| `model_size` | (Optional) Whisper model size to use (`small`, `medium`, `large-v2`, `large-v3`). Larger models require more RAM (default: `medium` requires ~5GB RAM)|


## Batch Transcription (all `.mp3` files):
To transcribe all mp3 files from the command prompt inside the container:
```bash
for f in audio-files/*.mp3; do
  python transcribe_sentences.py "$f" audio-files
done
```

## 📚 Why Use This?
When learning a new language, especially through podcasts, having accurate, aligned transcriptions is essential for comprehension and retention. Many language learning apps impose monthly transcription limits or rely on cloud-based AI. This tool gives you full control over your data, with no recurring costs, and the power of Whisper—all on your own hardware.

## 📦 Output
Transcriptions are saved in sentence-separated `.txt` format, ready for import into language learning platforms.

---

## 📝 License

This project is licensed under the [MIT License](LICENSE).