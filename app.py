import os
import streamlit as st
from streamlit_mic_recorder import speech_to_text
from streamlit_TTS import auto_play, text_to_speech, text_to_audio
import speech_recognition as sr
from gtts import gTTS
import tempfile
from langdetect import detect
import sounddevice as sd
import numpy as np
from pydub import AudioSegment
from openai import OpenAI

# Initialize OpenAI client
client = OpenAI(
    api_key=st.secrets["API_KEY"],  # Replace this with your API key
    base_url="https://api.aimlapi.com",
)

# Maintain a conversation history and user profiles
conversation_history = []
user_profile = {"language": "en", "preferences": {}, "age": None, "weight": None}

# Sample medical condition checker (mock integration with medical databases)
medical_conditions_db = {
    "fever": "You might be experiencing a viral infection or flu. Please monitor your temperature and stay hydrated. If it persists, consult a healthcare professional.",
    "headache": "A headache could be due to stress, dehydration, or lack of sleep. Rest and drink plenty of fluids. If it lasts longer than a few days, see a doctor.",
}

# Supported language mappings for SpeechRecognition and gTTS
language_mapping = {
    "Afrikaans": "af", "Amharic": "am", "Arabic": "ar", "Bulgarian": "bg",
    "Bengali": "bn", "Bosnian": "bs", "Catalan": "ca", "Czech": "cs",
    "Welsh": "cy", "Danish": "da", "German": "de", "Greek": "el", 
    "English": "en", "Spanish": "es", "Estonian": "et", "Basque": "eu",
    "Finnish": "fi", "French": "fr", "Galician": "gl", "Gujarati": "gu",
    "Hausa": "ha", "Hindi": "hi", "Croatian": "hr", "Hungarian": "hu",
    "Indonesian": "id", "Icelandic": "is", "Italian": "it", "Hebrew": "iw",
    "Japanese": "ja", "Javanese": "jw", "Khmer": "km", "Kannada": "kn",
    "Korean": "ko", "Latin": "la", "Lithuanian": "lt", "Latvian": "lv",
    "Malayalam": "ml", "Marathi": "mr", "Malay": "ms", "Myanmar (Burmese)": "my",
    "Nepali": "ne", "Dutch": "nl", "Norwegian": "no", "Punjabi (Gurmukhi)": "pa",
    "Polish": "pl", "Tamil": "ta", "Telugu": "te", "Thai": "th", 
    "Filipino": "tl", "Turkish": "tr", "Ukrainian": "uk", "Urdu": "ur", 
    "Vietnamese": "vi", "Cantonese": "yue", "Chinese (Simplified)": "zh-CN",
    "Chinese (Mandarin/Taiwan)": "zh-TW", "Chinese (Mandarin)": "zh"
}

# Global variables for audio control
current_audio = None
audio_segment = None
audio_file = None

def detect_language(text):
    """Detect the language of the input text."""
    return detect(text)

def get_response(user_input, age=None, weight=None):
    """Get AI response from OpenAI model."""
    conversation_history.append({"role": "user", "content": user_input})
    
    system_message = "You are an AI assistant who knows everything."
    if age is not None and weight is not None:
        system_message += f" The user is {age} years old and weighs {weight} kg."
    
    response = client.chat.completions.create(
        model="gpt-4o-2024-08-06",
        messages=[{"role": "system", "content": system_message}] + conversation_history
    )
    
    assistant_message = response.choices[0].message.content
    st.write(f"Assistant: {assistant_message}")
    
    conversation_history.append({"role": "assistant", "content": assistant_message})
    
    return assistant_message

def check_symptom(symptom_description):
    """Check symptoms against a mock database."""
    for symptom, advice in medical_conditions_db.items():
        if symptom in symptom_description.lower():
            return advice
    return "I'm not sure about this symptom. Please consult a doctor for a more accurate diagnosis."

def single_input_interaction():
    st.write("Press the button below and speak.")
    
    # Use the speech_to_text function for recording and converting speech to text
    user_input = speech_to_text(
        language='en',
        start_prompt="Start recording",
        stop_prompt="Stop recording",
        just_once=True
    )
    
    if user_input:
        st.write(f"You: {user_input}")
        detected_lang = detect_language(user_input)
        lang = language_mapping.get(detected_lang, user_profile["language"])
        st.write(f"Detected language: {lang}")

        # Determine appropriate response
        response_text = check_symptom(user_input)
        if response_text == "I'm not sure about this symptom. Please consult a doctor for a more accurate diagnosis.":
            response_text = get_response(user_input, user_profile["age"], user_profile["weight"])
        audio = text_to_audio(response_text, lang)
        auto_play(audio)

# Streamlit UI
st.title("Real-time Multilingual Audio Health Assistant")
st.sidebar.header("User Profile")
user_language = st.sidebar.selectbox("Select Your Language", list(language_mapping.keys()))
user_profile["language"] = language_mapping[user_language]
user_age = st.sidebar.number_input("Enter Your Age", min_value=0, max_value=120, value=30)
user_profile["age"] = user_age
user_weight = st.sidebar.number_input("Enter Your Weight (kg)", min_value=0, value=70)
user_profile["weight"] = user_weight

# Start interaction
single_input_interaction()
