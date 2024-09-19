
import librosa
import numpy as np
import soundfile as sf
import matplotlib.pyplot as plt
import IPython.display as ipd

# Load your audio file
file_path = '/content/Yeh Jism - Jism 2 128 Kbps.wav'  # Path to your uploaded file
audio_data, sample_rate = librosa.load(file_path, sr=None)

# Plot original audio waveform
plt.figure(figsize=(10, 4))
plt.plot(audio_data)
plt.title('Original Audio Waveform')
plt.show()

# Apply pitch shift
pitch_shift_steps = 4  # Adjust pitch: positive or negative for different results
pitch_shifted_audio = librosa.effects.pitch_shift(y=audio_data, sr=sample_rate, n_steps=pitch_shift_steps)

# Apply time stretching
time_stretch_rate = 1.5  # >1.0 speeds up, <1.0 slows down
time_stretched_audio = librosa.effects.time_stretch(audio_data, rate=time_stretch_rate)

# Apply dynamic range compression (amplitude modulation)
compressed_audio = librosa.effects.preemphasis(audio_data, coef=0.97)

# Apply equalization by boosting high frequencies
high_freq_boost = np.copy(audio_data)
high_freq_boost = high_freq_boost * np.linspace(1.0, 1.5, num=len(high_freq_boost))

# Save the modified audios to files
sf.write('pitch_shifted_audio.wav', pitch_shifted_audio, sample_rate)
sf.write('time_stretched_audio.wav', time_stretched_audio, sample_rate)
sf.write('compressed_audio.wav', compressed_audio, sample_rate)
sf.write('high_freq_boost_audio.wav', high_freq_boost, sample_rate)

# Plot and Play the original and modified audio
def plot_and_play(audio, title):
    plt.figure(figsize=(10, 4))
    plt.plot(audio)
    plt.title(title)
    plt.show()
    return ipd.Audio(audio, rate=sample_rate)

# Original Audio
print("Original Audio:")
ipd.display(ipd.Audio(audio_data, rate=sample_rate))
plot_and_play(audio_data, 'Original Audio Waveform')

# Pitch Shifted Audio
print("Pitch Shifted Audio:")
ipd.display(plot_and_play(pitch_shifted_audio, 'Pitch Shifted Audio Waveform'))

# Time Stretched Audio
print("Time Stretched Audio:")
ipd.display(plot_and_play(time_stretched_audio, 'Time Stretched Audio Waveform'))

# Compressed Audio
print("Compressed Audio (Dynamic Range Compression):")
ipd.display(plot_and_play(compressed_audio, 'Compressed Audio Waveform'))

# High Frequency Boost Audio
print("High Frequency Boosted Audio:")
ipd.display(plot_and_play(high_freq_boost, 'High Frequency Boost Audio Waveform'))

# Download the modified audio files in Colab
from google.colab import files
files.download('pitch_shifted_audio.wav')
files.download('time_stretched_audio.wav')
files.download('compressed_audio.wav')
files.download('high_freq_boost_audio.wav')
