from resemblyzer import VoiceEncoder, preprocess_wav
import numpy as np

encoder = VoiceEncoder()

wav = preprocess_wav("my_voice.wav")
embedding = encoder.embed_utterance(wav)

np.save("my_voice.npy", embedding)
print("Voice profile saved")
