import streamlit as st
import pyaudio
import wave
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
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
if GROQ_API_KEY is None:
    st.error("GROQ_API_KEY environment variable not set.")
    st.stop()

client = Groq(api_key=GROQ_API_KEY)

# Initialize text-to-speech engine
engine = pyttsx3.init()

# Function to record audio
def record_audio(duration=5):
    CHUNK = 1024
    FORMAT = pyaudio.paInt16
    CHANNELS = 2
    RATE = 44100

    p = pyaudio.PyAudio()
    stream = p.open(format=FORMAT,
                    channels=CHANNELS,
                    rate=RATE,
                    input=True,
                    frames_per_buffer=CHUNK)

    st.write("Recording...")
    frames = []
    for i in range(int(RATE / CHUNK * duration)):
        data = stream.read(CHUNK)
        frames.append(data)

    st.write("Recording complete.")
    stream.stop_stream()
    stream.close()
    p.terminate()

    # Create a temporary file
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.wav')
    temp_file_name = temp_file.name
    temp_file.close()

    wf = wave.open(temp_file_name, 'wb')
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(p.get_sample_size(FORMAT))
    wf.setframerate(RATE)
    wf.writeframes(b''.join(frames))
    wf.close()

    return temp_file_name

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
        except Exception as e:  
            if attempt < retry_attempts - 1:
                st.write(f"Error occurred: {e}. Retrying in 5 seconds...")
                time.sleep(5)  
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

def main():
    st.title("Voice Chatbot")

    if st.button("Record"):
        audio_file_name = record_audio()
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

if __name__ == "__main__":
    main()
