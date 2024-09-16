import streamlit as st
import sounddevice as sd
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
    st.write("Recording...")
    audio = sd.rec(int(duration * fs), samplerate=fs, channels=2, dtype='int16')
    sd.wait()
    st.write("Recording complete.")
    return audio, fs

# Function to save audio to a file
def save_audio_file(audio, fs, file_name):
    write(file_name, fs, audio)

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
    audio, fs = record_audio()
    with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as audio_file:
        audio_file_name = audio_file.name
    save_audio_file(audio, fs, audio_file_name)
    st.write("Audio recorded. Processing...")
    
    # Convert audio to text
    text = transcribe_audio(audio_file_name)
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
    os.remove(response_audio_file)
