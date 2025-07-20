import whisper
import nltk
import sys
import os

# Ensure NLTK punkt is available
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)

def transcribe_with_sentences(audio_file, output_dir, language, model_size):
    # Load Whisper model
    model = whisper.load_model(model_size)

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Transcribe audio
    result = model.transcribe(audio_file, language=language, verbose=True)
    text = result["text"]

    # Split text into sentences
    sentences = nltk.sent_tokenize(text)

    # Write sentences to output file with blank lines
    output_file = os.path.join(output_dir, os.path.basename(audio_file) + "_sentences.txt")
    with open(output_file, "w") as f:
        for sentence in sentences:
            f.write(f"{sentence.strip()}\n\n")

if __name__ == "__main__":
    if len(sys.argv) < 3 or len(sys.argv) > 5:
        print("Usage: python transcribe_sentences.py <audio_file> <output_dir> [language (default 'es')] [model_size (default 'medium')]")
        sys.exit(1)
    audio_file = sys.argv[1]
    output_dir = sys.argv[2]
    language = sys.argv[3] if len(sys.argv) >= 4 else "es"
    model_size = sys.argv[4] if len(sys.argv) == 5 else "medium"
    transcribe_with_sentences(audio_file, output_dir, language, model_size)