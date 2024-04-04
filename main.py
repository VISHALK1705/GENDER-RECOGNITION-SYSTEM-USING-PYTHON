# GENDER-RECOGNITION-SYSTEM-USING-PYTHON
import speech_recognition as sr
import numpy as np
import matplotlib.pyplot as plt

def record_audio(duration=5):
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        print("Recording...")
        audio_data = recognizer.listen(source, timeout=duration)
        print("Recording ended.")
    return audio_data

def extract_pitch(audio_data):
    audio_array = np.frombuffer(audio_data.frame_data, dtype=np.int16)
    pitch = np.mean(np.abs(np.diff(audio_array)))
    return pitch

def plot_audio_waveform(audio_data):
    audio_array = np.frombuffer(audio_data.frame_data, dtype=np.int16)
    duration = len(audio_array) / audio_data.sample_rate
    time = np.linspace(0, duration, len(audio_array))
    
    plt.figure(figsize=(10, 3))
    plt.plot(time, audio_array)
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.title("Audio Waveform")
    plt.show()

def predict_gender(pitch):
    if pitch > 250:
        return "female"
    else:
        return "male"

def main():
    audio_data = record_audio()
    pitch = extract_pitch(audio_data)
    plot_audio_waveform(audio_data)
    predicted_gender = predict_gender(pitch)
    print("Predicted gender:", predicted_gender)

if __name__ == "__main__":
    main()
