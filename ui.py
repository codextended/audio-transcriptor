import streamlit as st
from audio_recorder_streamlit import audio_recorder
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

client = Groq()

def transcribe_audio(audio_path):
    with open(audio_path, "rb") as file:
        transcription = client.audio.transcriptions.create(
            file=(audio_path, file.read()),
            model="whisper-large-v3-turbo",
            response_format="verbose_json",
        )
    return transcription.text

def fetch_ai_response(input_text):
    messages = [{'role': 'user', 'content': input_text}]
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=messages,
        temperature=0.5, 
        stream=True
    )
    return response.choices[0].message.content

def english_transcription(input_text):
    messages = [
        {'role': 'system', 'content':'You are a professional translator. Try your best to translate the following piece of text to "English". Just return the translated text with no furter explanation.'},
        {'role': 'user', 'content': input_text}
    ]
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=messages,
        temperature=0.5
    )
    return response.choices[0].message.content

def creole_transcription(input_text):
    messages = [
        {'role': 'system', 'content':'You are a professional translator. Try your best to translate the following piece of text to "Haitian Creole". Just return the translated text with no furter explanation.'},
        {'role': 'user', 'content': input_text}
    ]
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=messages,
        temperature=0.5
    )
    return response.choices[0].message.content

st.title("Creole-English Transcriptor")
st.write("Hi there! Click on the voice recorder to interact with me")

# recorded_audio = st.audio_input("Record your voice")

recorded_audio = audio_recorder()
if recorded_audio:
    # audio_file = "audio.mp3"
    # with open(audio_file, "wb") as f:
    #     f.write(recorded_audio)
    transcribed_text = transcribe_audio(recorded_audio)
    # st.write_stream(transcribed_text)
    st.write("Original transcription:\n", transcribed_text)

    english_transcript = english_transcription(transcribed_text)
    st.write("English transcrip:\n", english_transcript)

    creole_transcript = creole_transcription(transcribed_text)
    st.write("Creole transcrip:\n", creole_transcript)