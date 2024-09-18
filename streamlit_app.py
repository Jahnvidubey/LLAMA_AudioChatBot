import streamlit as st
import pyaudio
import wave
import pydub
import numpy as np
import pyttsx3
import tempfile
from scipy.io.wavfile import write
from whisper import load_model
from groq import Groq
import os
import time

# Initialize Whisper model
model = load_model("base")

# Initialize Groq client
client = Groq(
    api_key=os.environ.get("GROQ_API_KEY"),
)

# Initialize text-to-speech engine
engine = pyttsx3.init()

# Function to record audio
def record_audio(duration=5, fs=44100):
    chunk = 1024
    format = pyaudio.paInt16
    channels = 2

    audio = pyaudio.PyAudio()
    stream = audio.open(format=format,
                        channels=channels,
                        rate=fs,
                        input=True,
                        frames_per_buffer=chunk)

    frames = []
    st.write("Recording...")
    for i in range(int(fs / chunk * duration)):
        data = stream.read(chunk)
        frames.append(data)

    stream.stop_stream()
    stream.close()
    audio.terminate()
    st.write("Recording complete.")

    wf = wave.open("temp.wav", 'wb')
    wf.setnchannels(channels)
    wf.setsampwidth(audio.get_sample_size(format))
    wf.setframerate(fs)
    wf.writeframes(b''.join(frames))
    wf.close()

    return "temp.wav", fs

# Function to save audio to a file
def save_audio_file(audio_file_name, new_file_name):
    sound = pydub.AudioSegment.from_wav(audio_file_name)
    sound.export(new_file_name, format="wav")

# Function to convert speech to text
def transcribe_audio(file_name):
    result = model.transcribe(file_name)
    return result["text"]

# Function to generate chat responses with retry logic
def generate_chat_response(prompt):
    retry_attempts = 3
    for attempt in range(retry_attempts):
        try:
            chat_completion = client.chat.completions.create(
                messages=[{"role": "user", "content": prompt}],
                model="llama-3.1-70b-versatile",
            )
            return chat_completion.choices[0].message.content
        except Exception as e:  # Catch all exceptions
            if attempt < retry_attempts - 1:
                st.write(f"Error occurred: {e}. Retrying in 5 seconds...")
                time.sleep(5)  # Wait before retrying
            else:
                st.write(f"Failed after {retry_attempts} attempts. Error: {e}")
                raise

# Function to convert text to speech
def text_to_speech(text):
    with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp_file:
        tmp_file_name = tmp_file.name
    engine.save_to_file(text, tmp_file_name)
    engine.runAndWait()
    return tmp_file_name

st.title("Voice Chatbot")

if st.button("Record"):
    audio_file_name, fs = record_audio()
    new_file_name = "recorded_audio.wav"
    save_audio_file(audio_file_name, new_file_name)
    st.write("Audio recorded. Processing...")
    
    # Convert audio to text
    text = transcribe_audio(new_file_name)
    st.write(f"Transcribed text: {text}")
    
    # Get response from LLAMA model
    response_text = generate_chat_response(text)
    st.write(f"LLAMA response: {response_text}")
    
    # Convert response to speech
    response_audio_file = text_to_speech(response_text)
    st.write("Response generated. Playing audio...")
    with open(response_audio_file, 'rb') as audio_file:
        audio_bytes = audio_file.read()
        st.audio(audio_bytes, format='audio/wav')

    # Clean up temporary files
    os.remove(audio_file_name)  
    os.remove(new_file_name)  
    os.remove(response_audio_file)
