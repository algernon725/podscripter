import whisper
import nltk
import sys
import os

# Ensure NLTK punkt is available
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)

def write_txt(sentences, output_file):
    try:
        with open(output_file, "w") as f:
            for sentence in sentences:
                f.write(f"{sentence.strip()}\n\n")
    except Exception as e:
        print(f"Error writing TXT file: {e}")
        sys.exit(1)

def write_srt(segments, output_file):
    def format_timestamp(seconds):
        h = int(seconds // 3600)
        m = int((seconds % 3600) // 60)
        s = int(seconds % 60)
        ms = int((seconds - int(seconds)) * 1000)
        return f"{h:02}:{m:02}:{s:02},{ms:03}"

    try:
        with open(output_file, "w") as f:
            for i, seg in enumerate(segments, 1):
                start = format_timestamp(seg['start'])
                end = format_timestamp(seg['end'])
                text = seg['text'].strip()
                f.write(f"{i}\n{start} --> {end}\n{text}\n\n")
    except Exception as e:
        print(f"Error writing SRT file: {e}")
        sys.exit(1)

def transcribe_with_sentences(audio_file, output_dir, language, model_size, output_format):
    # Load Whisper model
    try:
        model = whisper.load_model(model_size)
    except Exception as e:
        print(f"Error loading Whisper model: {e}")
        sys.exit(1)

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Transcribe audio
    try:
        result = model.transcribe(audio_file, language=language, verbose=True)
    except Exception as e:
        print(f"Error during transcription: {e}")
        sys.exit(1)

    text = result["text"]
    base_name = os.path.splitext(os.path.basename(audio_file))[0]  # Strip extension

    if output_format == "srt":
        output_file = os.path.join(output_dir, base_name + ".srt")
        write_srt(result["segments"], output_file)
    else:
        # Split text into sentences
        sentences = nltk.sent_tokenize(text) # Use NLTK to split into sentences
        output_file = os.path.join(output_dir, base_name + ".txt")
        write_txt(sentences, output_file)

if __name__ == "__main__":
    if len(sys.argv) < 3 or len(sys.argv) > 6:
        print("Usage: python transcribe_sentences.py <audio_file> <output_dir> [language (default 'en')] [model_size (default 'medium')] [output_format (txt|srt, default 'txt')]")
        sys.exit(1)
    audio_file = sys.argv[1]
    output_dir = sys.argv[2]
    language = sys.argv[3] if len(sys.argv) >= 4 else "en"
    model_size = sys.argv[4] if len(sys.argv) >= 5 else "medium"
    output_format = sys.argv[5] if len(sys.argv) == 6 else "txt"
    transcribe_with_sentences(audio_file, output_dir, language, model_size, output_format)