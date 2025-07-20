import whisper
import nltk
import sys
import os

# Ensure NLTK punkt is available
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)

# Load Whisper model
model = whisper.load_model("medium")

def transcribe_with_sentences(audio_file, output_dir):
    # Transcribe audio
    result = model.transcribe(audio_file, language="es")
    text = result["text"]

    # Split text into sentences
    sentences = nltk.sent_tokenize(text)

    # Write sentences to output file with blank lines
    output_file = os.path.join(output_dir, os.path.basename(audio_file) + "_sentences.txt")
    with open(output_file, "w") as f:
        for sentence in sentences:
            f.write(f"{sentence.strip()}\n\n")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python transcribe_sentences.py <audio_file> <output_dir>")
        sys.exit(1)
    audio_file = sys.argv[1]
    output_dir = sys.argv[2]
    transcribe_with_sentences(audio_file, output_dir)