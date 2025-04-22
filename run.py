import pyaudio
import numpy as np
import wave
import openwakeword
from openwakeword.model import Model
import time
import sys
import subprocess
import requests
from collections import deque

# === Konfiguration ===
API_KEY = ""
API_URL = "https://chat-ai.academiccloud.de/v1/chat/completions"
MODEL_NAME = "meta-llama-3.1-8b-instruct"

# Audio
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000
CHUNK = 1280
PRE_BUFFER_SECONDS = 0.5
MAX_SILENCE_TIME = 1.5
RECORDING_TIMEOUT = 12

# Schwellen
VOLUME_THRESHOLD = 900
THRESHOLD = 0.985
RESET_THRESHOLD = 0.4

# Abgeleitete Werte
max_silence_chunks = int(MAX_SILENCE_TIME * RATE / CHUNK)
pre_buffer_chunks = int(PRE_BUFFER_SECONDS * RATE / CHUNK)

# Wakeword-Modell laden
openwakeword.utils.download_models()
oww_model = Model(wakeword_models=["alexa"])

wakeword_active = True
last_trigger = 0

# === Hilfsfunktionen ===

def play_beep(frequency=1000, duration_ms=200, volume=0.2, output_device_index=0):
    fs = 44100
    t = np.linspace(0, duration_ms / 1000, int(fs * duration_ms / 1000), False)
    tone = np.sin(frequency * 2 * np.pi * t)
    audio_data = (tone * volume * 32767).astype(np.int16).tobytes()

    p = pyaudio.PyAudio()
    try:
        stream = p.open(format=pyaudio.paInt16, channels=1, rate=fs, output=True, output_device_index=output_device_index)
        stream.write(audio_data)
        stream.stop_stream()
        stream.close()
    finally:
        p.terminate()

def transkribiere_whisper(wav_path):
    subprocess.run([
       "/home/pi/whisper.cpp/build/bin/whisper-cli",
        "-m", "/home/pi/whisper.cpp/models/ggml-tiny.bin",
        "-f", wav_path,
        "-l", "de",
        "-otxt"
    ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        with open(wav_path + ".txt", "r") as f:
            text = f.read().strip()
        print("📝 Transkription:", text)
        return text
    except FileNotFoundError:
        print("⚠️ Keine Transkriptionsdatei gefunden.")
        return ""

def frage_llama3(prompt):
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }
    data = {
        "model": MODEL_NAME,
        "messages": [
            {"role": "system", "content": "Du bist ein hilfreicher Assistent."},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.7
    }
    try:
        response = requests.post(API_URL, json=data, headers=headers, timeout=60)
        response.raise_for_status()
        reply = response.json()["choices"][0]["message"]["content"]
        print("💬 Antwort:", reply)
        return reply
    except Exception as e:
        print("❌ Fehler bei der Anfrage:", e)
        return "[Fehler bei der Anfrage]"

def record_after_wakeword(stream):
    print("📼 Warte auf Sprache...")
    pre_buffer = deque(maxlen=pre_buffer_chunks)
    recording = []
    silence_count = 0
    total_chunks = int(RECORDING_TIMEOUT * RATE / CHUNK)
    speech_detected = False

    for _ in range(pre_buffer_chunks):
        data = stream.read(CHUNK, exception_on_overflow=False)
        pre_buffer.append(data)

    play_beep(frequency=1000, duration_ms=200)
    time.sleep(0.08)
    for _ in range(int((200 / 1000) * RATE / CHUNK) + 5):
        _ = stream.read(CHUNK, exception_on_overflow=False)
    time.sleep(0.05)
    print("🎙️ Aufnahme läuft...")

    for i in range(total_chunks):
        data = stream.read(CHUNK, exception_on_overflow=False)
        audio_np = np.frombuffer(data, dtype=np.int16)
        volume = np.sqrt(np.mean(audio_np.astype(np.float32) ** 2))

        recording.append(data)
        if volume > VOLUME_THRESHOLD:
            silence_count = 0
            speech_detected = True
        else:
            silence_count += 1

        if speech_detected and silence_count > max_silence_chunks:
            print("🤫 Stille erkannt – Aufnahme beendet.")
            break

    return list(pre_buffer) + recording, speech_detected

# === Hauptprogramm ===

p = pyaudio.PyAudio()
stream = p.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK)
print("🎤 Mikrofon gestartet. Warte auf Hotword... ('Alexa')")

try:
    while True:
        audio_chunk = stream.read(CHUNK, exception_on_overflow=False)
        audio_np = np.frombuffer(audio_chunk, dtype=np.int16)
        prediction = oww_model.predict(audio_np)
        conf = prediction["alexa"]

        if conf > THRESHOLD and wakeword_active:
            print("🎉 Wakeword erkannt!")
            wakeword_active = False
            last_trigger = time.time()

            frames, speech_detected = record_after_wakeword(stream)
            play_beep(frequency=600, duration_ms=200)

            if speech_detected:
                wav_path = "aufnahme.wav"
                with wave.open(wav_path, 'wb') as wf:
                    wf.setnchannels(CHANNELS)
                    wf.setsampwidth(pyaudio.get_sample_size(FORMAT))
                    wf.setframerate(RATE)
                    wf.writeframes(b''.join(frames))
                print("✅ Aufnahme gespeichert als 'aufnahme.wav'")

                # ▶️ Wiedergabe
                wf = wave.open(wav_path, 'rb')
                p_play = pyaudio.PyAudio()
                stream_out = p_play.open(
                    format=p_play.get_format_from_width(wf.getsampwidth()),
                    channels=wf.getnchannels(),
                    rate=wf.getframerate(),
                    output=True
                )
                data = wf.readframes(CHUNK)
                while data:
                    stream_out.write(data)
                    data = wf.readframes(CHUNK)
                stream_out.stop_stream()
                stream_out.close()
                wf.close()
                p_play.terminate()

                # 💬 Transkription + GPT
                text = transkribiere_whisper(wav_path)
                if text:
                    antwort = frage_llama3(text)
                    print("📣", antwort)
            else:
                print("🔇 Keine Sprache erkannt – verworfen.")

        elif conf < RESET_THRESHOLD and not wakeword_active and (time.time() - last_trigger > 2):
            wakeword_active = True
            print("🟢 Wakeword wieder freigegeben")

except KeyboardInterrupt:
    print("👋 Beende auf Benutzerwunsch...")

finally:
    try:
        if stream.is_active():
            stream.stop_stream()
        stream.close()
        p.terminate()
        print("🧼 Aufgeräumt. Tschüss!")
    except Exception as e:
        print("⚠️ Fehler beim Aufräumen:", e)
