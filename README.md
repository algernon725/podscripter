# whisper-docker-nltk
---

## Pre-requisites:

1. An Apple Mac computer with M1, M2, M3, or M4 processor
2. Install [Docker](https://www.docker.com)

## Setup Instructions

1. Create these 3  folders on your local drive:
```bash
mkdir whisper-docker-nltk
mkdir whisper-docker-nltk/audio-files
mkdir whisper-docker-nltk/models
```

2. Navigate to the whisper-docker-nltk directory:
```bash
cd whisper-docker-nltk
```

3. Build the Docker Image
```bash
docker build --platform linux/arm64 -t whisper-ai-nltk .
```

4. Run the Docker Container
```bash
docker run --platform linux/arm64 -it -v $(pwd)/models:/root/.cache/whisper -v $(pwd)/audio-files:/app/audio-files whisper-ai-nltk
```

## Usage Instructions

Start transcribing using Whisper by executing this script inside the Container
Usage: python transcribe_sentences.py <audio_file> <output_dir> [language (default 'es')] [model_size (default 'medium')]"
```bash
python transcribe_sentences.py audio-files/test.mp3 audio-files
```

How to batch process all mp3 files in the audio-files folder:
You need to loop over the files in a shell script or in Python.
For example, in your terminal (bash/zsh):
```bash
for f in audio-files/*.mp3; do
    python transcribe_sentences.py "$f" audio-files
done
```

## Command line arguments:
| Argument   | Description |
| ---------- | ----------- |
| audio_file | audio file to be transcribed |
| output_dir | Directory to save the transcription output |
| language   | Specifies the language of the transcribed output file (es is the default) |
| model_size | Specifies the Whisper model size. Options include small, medium, large-v2, large-v3 (larger models require more memory) |