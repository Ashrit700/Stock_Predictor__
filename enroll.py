import sounddevice as sd
from scipy.io.wavfile import write

fs = 16000
seconds = 10

print("Speak naturally for 10 seconds...")
audio = sd.rec(int(seconds * fs), samplerate=fs, channels=1)
sd.wait()

write("my_voice.wav", fs, audio)
print("Voice enrolled")
