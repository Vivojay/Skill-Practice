# From openai (local model)
import whisper

import openai

import sounddevice as sd
import numpy as np
from scipy.io.wavfile import write
from pydub import AudioSegment
import os

# Load the desired model
model_name = "base"  # Choose from tiny, base, small, medium, large
model = whisper.load_model(model_name)

# Transcribe an audio file
def transcribe_audio(audio_path):
    result = model.transcribe(audio_path)
    return result['text']

def is_silence(audio_chunk, silence_threshold=0.01):
    """Check if the audio chunk is silent based on a threshold."""
    return np.abs(audio_chunk).mean() < silence_threshold

def record_until_pause(filename="./tmp/usr_aud.mp3", silence_duration=2000, sample_rate=16000):

    buffer_time = 1000
    buffer_time = silence_duration//1000
    silence_duration = round(silence_duration/1000, 2)
    # print(f"Recording... Speak now! (Stops when you pause for more than {silence_duration} seconds)")
    print('listening...')

    silence_threshold = 0.01  # Adjust for sensitivity (lower = more sensitive to noise)

    feedback_limit = int(silence_duration * sample_rate)
    silence_limit = int((silence_duration + buffer_time) * sample_rate)
    feedback_printed = False
    buffer = []
    silent_samples = 0

    # Stream audio from microphone
    with sd.InputStream(samplerate=sample_rate, channels=1, dtype='float32') as stream:
        while True:
            # Read audio in small chunks
            audio_chunk = stream.read(1024)[0]
            buffer.append(audio_chunk)
            
            # Check for silence
            if is_silence(audio_chunk, silence_threshold):
                silent_samples += len(audio_chunk)

                if not feedback_printed and (silent_samples > feedback_limit):
                    print("Detected silence. Stopping recording.")
                    feedback_printed = True

                if silent_samples > silence_limit:
                    break
            else:
                silent_samples = 0  # Reset silent sample counter on activity

    if not os.path.isdir('./tmp'):
        os.mkdir('./tmp')

    # Save the recorded audio
    # print("Processing audio...")
    audio_data = np.concatenate(buffer, axis=0)
    wav_filename = "./tmp/temp_output.wav"
    write(wav_filename, sample_rate, (audio_data * 32767).astype(np.int16))  # Convert to WAV

    # Convert WAV to MP3
    # print("Converting to MP3...")
    audio = AudioSegment.from_wav(wav_filename)
    audio.export(filename, format="mp3")
    # print(f"Audio saved as {filename}")

    # Cleanup temporary files
    os.remove(wav_filename)

    return filename

"""
# Sample tested code for training and creating a model for chess text to notation in:
./model_train.py
"""

from transformers import T5ForConditionalGeneration, T5Tokenizer

def decode(model, tokenizer, input_text):
    # Tokenize the input
    input_encodings = tokenizer(input_text, padding=True, truncation=True, return_tensors="pt", max_length=128)
    # Generate the output using the model
    output = model.generate(input_encodings['input_ids'], max_length=128, num_beams=4, early_stopping=True)
    # Decode the generated tokens into text
    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
    return generated_text

# Example input
input_texts = [
    "move the knight from c4 to d6",
    "king from c3 to d6",
    "move knight from c3 to d6",
    "move knight from c2 to e6",
    "Move night from c1 to h3",
    "Move night from d4 to f6",
    "king from c4 to h6",
    "rook h6 to a4",
]


tokenizer_path = f"./results/checkpoint-600"
def test_input(model, input_texts, checkpoint):
    checkpoint_path = f"./results/checkpoint-{checkpoint}"
    # Load the trained model and tokenizer
    model = T5ForConditionalGeneration.from_pretrained(checkpoint_path)
    tokenizer = T5Tokenizer.from_pretrained(tokenizer_path)  # or your specific tokenizer
    res = [decode(model, tokenizer, input_text) for input_text in input_texts]
    return res

test_input(model, input_texts, checkpoint=500)

def rule_based(input_text):
    input_text = input_text.lower()
    remove_words = 'left right moves move goes go captures capture slides slide on at it the from to'.split()
    for rm_word in remove_words:
        input_text = input_text.replace(rm_word, '')
    print(input_text)
    p, i, f = input_text.split()
    if p == 'knight' or 'night':
        p = 'N'
    else:
        p = p[0].upper()
    return f"{p}({i}){f}"

"""
# Example usage: Provide the path to your audio file (e.g., .mp3, .wav)
audio_file = "./tmp/usr_aud.mp3"
transcription = transcribe_audio(audio_file)
"""

