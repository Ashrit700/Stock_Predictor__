import speech_recognition as sr
import pyttsx3
import os
import webbrowser
import time
import re
import urllib.parse
import numpy as np
from scipy.io.wavfile import write
from resemblyzer import VoiceEncoder, preprocess_wav

# ================= VOICE AUTH =================
encoder = VoiceEncoder()
stored_embedding = np.load("my_voice.npy")

def record_temp_voice(filename="temp.wav", seconds=3):
    import sounddevice as sd
    fs = 16000
    audio = sd.rec(int(seconds * fs), samplerate=fs, channels=1)
    sd.wait()
    write(filename, fs, audio)

def is_my_voice(audio_file):
    wav = preprocess_wav(audio_file)
    new_embedding = encoder.embed_utterance(wav)
    similarity = np.dot(stored_embedding, new_embedding)
    return similarity > 0.75

# ================= TEXT TO SPEECH =================
engine = pyttsx3.init("sapi5")
engine.setProperty("rate", 145)
engine.setProperty("volume", 1.0)

def speak(text):
    print("Jarvis:", text)
    engine.say(text)
    engine.runAndWait()
    time.sleep(1)   # 🔴 REQUIRED DELAY

# ================= SPEECH TO TEXT =================
recognizer = sr.Recognizer()

def listen():
    with sr.Microphone() as source:
        recognizer.adjust_for_ambient_noise(source, duration=0.6)
        audio = recognizer.listen(source)
    try:
        return recognizer.recognize_google(audio).lower()
    except:
        return ""

# ================= CLEAN INPUT =================
def clean(text):
    fillers = [
        "please", "can you", "could you", "help me",
        "i want", "i need", "for me", "jarvis"
    ]
    for f in fillers:
        text = text.replace(f, "")
    return text.strip()

# ================= COMMAND HANDLER =================
def handle_command(text):
    text = clean(text)

    if "youtube" in text:
        query = re.sub(r"(open|search|play|find|youtube|on)", "", text).strip()
        if query:
            q = urllib.parse.quote(query)
            webbrowser.open(f"https://www.youtube.com/results?search_query={q}")
            return f"Searching {query} on YouTube"
        webbrowser.open("https://www.youtube.com")
        return "Opening YouTube"

    if "boss" in text or "chatgpt" in text:
        query = re.sub(r"(search|ask|boss|chatgpt|about|on)", "", text).strip()
        if query:
            q = urllib.parse.quote(query)
            webbrowser.open(f"https://chat.openai.com/?q={q}")
            return f"Asking boss about {query}"
        webbrowser.open("https://chat.openai.com")
        return "Opening boss"

    if "chrome" in text or "browser" in text:
        os.system("start chrome")
        return "Opening Chrome"

    if "file" in text or "explorer" in text:
        os.system("explorer")
        return "Opening File Explorer"

    if "notepad" in text:
        os.system("notepad")
        return "Opening Notepad"    

    if "wordpad" in text:
        os.system("write")
        return "Opening WordPad"

    if "eclipse" in text:
        os.startfile(r"C:\eclipse\eclipse.exe")  # 🔴 UPDATE PATH
        return "Opening Eclipse IDE"

    if "gmail" in text or "mail" in text:
        webbrowser.open("https://mail.google.com")
        return "Opening Gmail"

    return "Sorry sir, I did not understand"

# ================= MAIN LOOP =================
speak("Jarvis is online")

while True:
    heard = listen()

    if "bye jarvis" in heard:
        speak("Goodbye sir")
        break

    if "jarvis" in heard:
        speak("Yes sir")

        # 🔐 VOICE VERIFICATION
        speak("Verifying voice")
        record_temp_voice()

        if not is_my_voice("temp.wav"):
            speak("Unauthorized voice detected")
            continue

        command = listen()
        if command:
            response = handle_command(command)
            speak(response)
