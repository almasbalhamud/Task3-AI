import whisper
import cohere
from gtts import gTTS
import os
import sounddevice as sd
from scipy.io.wavfile import write
import numpy as np
import tkinter as tk
import playsound

co = cohere.Client("API code")
model = whisper.load_model("base")

def generate_response(user_input):
    response = co.generate(
        model="command",  
        prompt=user_input,
        max_tokens=100,
        temperature=0.7
    )
    return response.generations[0].text.strip()

def start_process():
    fs = 44100  
    duration = 5  

    recording = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype='int16')
    sd.wait()
    write("live_record.wav", fs, recording)

    result = model.transcribe("live_record.wav")
    print(result["text"])

    text_from_voice = result["text"]
    reply = generate_response(text_from_voice)
    print(reply)

    tts = gTTS(text=reply, lang='ar')
    tts.save("response.mp3")
    playsound.playsound("response.mp3")

root = tk.Tk()
root.title("تشغيل المساعد الصوتي")
root.geometry("300x150")

button = tk.Button(root, text="ابدأ التسجيل", font=("Arial", 14), command=start_process)
button.pack(pady=40)

root.mainloop()
