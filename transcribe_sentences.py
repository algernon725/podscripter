import nltk
import sys
import os
import glob
import time

#import whisper
from pydub import AudioSegment
from faster_whisper import WhisperModel

# Ensure NLTK punkt is available
#nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)

def split_audio(audio_file, chunk_length_sec=300):
    audio = AudioSegment.from_file(audio_file)
    chunks = []
    for i in range(0, len(audio), chunk_length_sec * 1000):
        chunk = audio[i:i + chunk_length_sec * 1000]
        chunk_path = f"{audio_file}_chunk_{i // (chunk_length_sec * 1000)}.wav"
        chunk.export(chunk_path, format="wav")
        chunks.append(chunk_path)
    return chunks

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
    os.environ["OMP_NUM_THREADS"] = "8"
    # Load faster-whisper model
    try:
        model = WhisperModel(model_size, device="cpu", compute_type="int8")  # or "cuda" for GPU
        #model = WhisperModel("turbo", device="cpu", compute_type="int8")  # or "cuda" for GPU
    except Exception as e:
        print(f"Error loading faster-whisper model: {e}")
        sys.exit(1)

    os.makedirs(output_dir, exist_ok=True)
    print("Splitting audio into chunks...")
    chunk_files = split_audio(audio_file, chunk_length_sec=300)  # 5 minute chunk size

    all_text = ""
    all_segments = []
    offset = 0.0

    print("Transcribing chunks...")
    for idx, chunk_file in enumerate(chunk_files, 1):
        print(f"Transcribing chunk {idx}/{len(chunk_files)}: {chunk_file}")
        try:
            segments, info = model.transcribe(chunk_file, language=language, beam_size=3)
            text = ""
            chunk_segments = []
            for seg in segments:
                seg_dict = {
                    "start": seg.start + offset,
                    "end": seg.end + offset,
                    "text": seg.text
                }
                chunk_segments.append(seg_dict)
                text += seg.text + " "
            all_segments.extend(chunk_segments)
            all_text += text.strip() + "\n"
            offset += AudioSegment.from_file(chunk_file).duration_seconds
        except Exception as e:
            print(f"Error during transcription: {e}")
            sys.exit(1)
        finally:
            if os.path.exists(chunk_file):
                os.remove(chunk_file)

    base_name = os.path.splitext(os.path.basename(audio_file))[0]
    if output_format == "srt":
        output_file = os.path.join(output_dir, base_name + ".srt")
        write_srt(all_segments, output_file)
    else:
        lang_map = {
            'en': 'english',
            'es': 'spanish',
            'de': 'german',
            'fr': 'french'
        }
        nltk_lang = lang_map.get(language, 'english')
        sentences = nltk.sent_tokenize(all_text, language=nltk_lang)
        output_file = os.path.join(output_dir, base_name + ".txt")
        write_txt(sentences, output_file)

def cleanup_chunks(audio_file):
    audio_dir = os.path.dirname(os.path.abspath(audio_file))
    pattern = os.path.join(audio_dir, "*_chunk_*.wav")
    for chunk_path in glob.glob(pattern):
        try:
            os.remove(chunk_path)
            print(f"Removed leftover chunk file: {chunk_path}")
        except Exception as e:
            print(f"Error removing chunk file {chunk_path}: {e}")

if __name__ == "__main__":
    if len(sys.argv) < 3 or len(sys.argv) > 6:
        print("Usage: python transcribe_sentences.py <audio_file> <output_dir> [language (default 'en')] [model_size (default 'medium')] [output_format (txt|srt, default 'txt')]")
        sys.exit(1)
    audio_file = sys.argv[1]
    output_dir = sys.argv[2]
    language = sys.argv[3] if len(sys.argv) >= 4 else "en"
    model_size = sys.argv[4] if len(sys.argv) >= 5 else "medium"
    output_format = sys.argv[5] if len(sys.argv) == 6 else "txt"
    if output_format not in ("txt", "srt"):
        print(f"Warning: Output format '{output_format}' not recognized. Defaulting to 'txt'.")
        output_format = "txt"    
    
    start_time = time.time()

    cleanup_chunks(audio_file)  # Cleanup any existing chunks before starting

    transcribe_with_sentences(audio_file, output_dir, language, model_size, output_format)

    end_time = time.time()
    elapsed = end_time - start_time
    minutes = int(elapsed // 60)
    seconds = int(elapsed % 60)
    print(f"Script completed in {minutes} minutes and {seconds} seconds.")