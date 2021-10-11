# script to test spectrograms

import librosa
import matplotlib
matplotlib.use('Agg') # No pictures displayed 
import librosa.display
import pylab
import numpy as np
import os
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import cv2
import spectrogram_visualizer

wav_file_path = "/Users/prakrutigogia/Documents/Microsoft/AlwaysBeLearning/MSHack/PodCast/Round2_OS_07_05/wav/1562337136_001a.wav"
# wav_file_path = "./wav_dir/2020-07-28T04_13_14.029002Z.wav"
spectrogram_visualizer.write_spectrogram(wav_file_path)

# y, sr = librosa.load(wav_file_path)
# print(y.shape)
# leny = len(y)//2
# X = librosa.stft(y[:leny])
# Xdb = librosa.amplitude_to_db(abs(X))

# spec_output_path = "./thing2.png"
# S = librosa.feature.melspectrogram(y=y[:leny], sr=sr, n_mels=128, fmax=10000)
# S_dB = librosa.power_to_db(S, ref=np.max)

# plt.figure()
# plt.get_current_fig_manager().full_screen_toggle()
# pylab.axis('off') # no axis
# pylab.axes([0., 0., 1., 1.], frameon=False, xticks=[], yticks=[]) # Remove the white edge
# librosa.display.specshow(Xdb, sr=sr, x_axis='time', y_axis='hz')
# pylab.savefig(spec_output_path, bbox_inches=None, pad_inches=0)

# fig, (ax1, ax2) = plt.subplots(nrows=2)
# ax1.plot(t, x)
# Pxx, freqs, bins, im = ax2.specgram(y, NFFT=512, Fs=sr, noverlap=10)
# plt.colorbar()
# plt.show()

# pylab.axis('off') # no axis
# pylab.axes([0., 0., 1., 1.], frameon=False, xticks=[], yticks=[]) # Remove the white edge
# librosa.display.specshow(S_dB, x_axis='time',y_axis='mel', sr=sr, fmax=8000)
# pylab.savefig(spec_output_path, bbox_inches=None, pad_inches=0)
