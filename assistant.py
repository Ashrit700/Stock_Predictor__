import speech_recognition as sr
import pyttsx3
import os
import webbrowser
import time
import urllib.parse
import re

# ================== TTS ==================
engine = pyttsx3.init("sapi5")
engine.setProperty("rate", 145)
engine.setProperty("volume", 1.0)

def speak(text):
    print("Jarvis:", text)
    engine.say(text)
    engine.runAndWait()
    time.sleep(1)   # ✅ REQUIRED DELAY BEFORE LISTENING AGAIN

# ================== STT ==================
recognizer = sr.Recognizer()

def listen():
    with sr.Microphone() as source:
        recognizer.adjust_for_ambient_noise(source, duration=0.6)
        audio = recognizer.listen(source)
    try:
        text = recognizer.recognize_google(audio)
        print("You:", text)
        return text.lower()
    except:
        return ""

# ================== CLEAN TEXT ==================
def clean(text):
    junk = [
        "please", "can you", "could you", "help me",
        "i want", "i need", "for me", "jarvis"
    ]
    for j in junk:
        text = text.replace(j, "")
    return text.strip()

# ================== COMMAND HANDLER ==================
def handle_command(text):
    text = clean(text)

    # ---- YOUTUBE ----
    if "youtube" in text:
        query = re.sub(r"(open|search|play|find|youtube|on)", "", text).strip()
        if query:
            q = urllib.parse.quote(query)
            webbrowser.open(
                f"https://www.youtube.com/results?search_query={q}"
            )
            return f"Searching {query} on YouTube"
        else:
            webbrowser.open("https://www.youtube.com")
            return "Opening YouTube"

    # ---- CHATGPT / BOSS ----
    if "boss" in text or "chatgpt" in text:
        query = re.sub(r"(search|ask|boss|chatgpt|about|on)", "", text).strip()
        if query:
            q = urllib.parse.quote(query)
            webbrowser.open(f"https://chat.openai.com/?q={q}")
            return f"Asking boss about {query}"
        else:
            webbrowser.open("https://chat.openai.com")
            return "Opening boss"

    # ---- CHROME ----
    if "chrome" in text or "browser" in text:
        os.system("start chrome")
        return "Opening Chrome"

    # ---- FILE EXPLORER ----
    if "file" in text or "explorer" in text:
        os.system("explorer")
        return "Opening File Explorer"

    # ---- NOTEPAD ----
    if "notepad" in text:
        os.system("notepad")
        return "Opening Notepad"

    # ---- WORDPAD ----
    if "wordpad" in text:
        os.system("write")
        return "Opening WordPad"

    # ---- ECLIPSE ----
    if "eclipse" in text:
        os.startfile(r"C:\eclipse\eclipse.exe")  # update path
        return "Opening Eclipse IDE"

    # ---- GMAIL ----
    if "gmail" in text or "mail" in text:
        webbrowser.open("https://mail.google.com")
        return "Opening Gmail"

    return "Sorry sir, I did not understand"

# ================== MAIN LOOP ==================
speak("Jarvis is online")

while True:
    heard = listen()

    if "thanks jarvis" in heard:
        speak("Goodbye sir")
        break

    if "jarvis" in heard:
        speak("Yes sir")
        command = listen()
        if command:
            response = handle_command(command)
            speak(response)
