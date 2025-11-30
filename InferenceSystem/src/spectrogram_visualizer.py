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

import json
import math


def _create_spectrogram_figure(specshow_data, sr, output_path, x_axis='time', y_axis='hz', fmax=None):
    """
    Helper function to create and save a spectrogram using explicit Figure/Axes objects.
    Closes the figure after saving to prevent memory leaks.
    """
    # Use explicit figure with size 6.4x4.8 inches at 100 dpi = 640x480 pixels
    fig = plt.figure(frameon=False, figsize=(6.4, 4.8), dpi=100)
    ax = fig.add_subplot(111)
    ax.axis('off')
    ax.set_position([0., 0., 1., 1.])  # Remove borders
    
    if fmax is not None:
        librosa.display.specshow(specshow_data, sr=sr, x_axis=x_axis, y_axis=y_axis, fmax=fmax, ax=ax)
    else:
        librosa.display.specshow(specshow_data, sr=sr, x_axis=x_axis, y_axis=y_axis, ax=ax)
    
    fig.savefig(output_path, bbox_inches=None, pad_inches=0)
    
    # Close figure to release memory and prevent leaks
    plt.close(fig)


def write_spectrogram(wav_file_path):
    """

    """
    
    # get wav_file_path without extension
    directory_name = os.path.dirname(wav_file_path)
    candidate_name = os.path.basename(wav_file_path)
    candidate_name_without_extension = os.path.splitext(candidate_name)[0]

    spectogram_name = candidate_name_without_extension + ".png"

    # temp files that will be deleted
    spec_first_half = os.path.join(directory_name, "firstHalf.png")
    spec_second_half = os.path.join(directory_name, "secondHalf.png")

    # final spec file
    spec_output_path = os.path.join(directory_name, spectogram_name)

    # Here, we divide the audio into spectrogram into 2 parts and calculate spectrograms for each half
    y, sr = librosa.load(wav_file_path)
    half_len_y = len(y)//2
    y_first_half = y[:half_len_y]
    y_second_half = y[half_len_y:]

    X_first_half = librosa.stft(y_first_half)
    Xdb_first_half = librosa.amplitude_to_db(abs(X_first_half))
    _create_spectrogram_figure(Xdb_first_half, sr, spec_first_half, x_axis='time', y_axis='hz')

    X_second_half = librosa.stft(y_second_half)
    Xdb_second_half = librosa.amplitude_to_db(abs(X_second_half))
    _create_spectrogram_figure(Xdb_second_half, sr, spec_second_half, x_axis='time', y_axis='hz')

    # create canvas to create combined spectrogram
    # Use dtype=np.uint8 to match images read by cv2
    canvas = np.zeros((480, 640*2, 3), dtype=np.uint8)

    # combine spectrograms
    spec1 = cv2.imread(spec_first_half)
    spec2 = cv2.imread(spec_second_half)

    # delete spec1 and spec2
    os.remove(spec_first_half)
    os.remove(spec_second_half)

    canvas[:, :640, :] = spec1
    canvas[:, 640:, :] = spec2

    cv2.imwrite(spec_output_path, canvas)

    return spec_output_path


def write_annotations_on_spectrogram(wav_file_path, wav_timestamp, data, spec_output_path):
    """

    """

    y, sr = librosa.load(wav_file_path)
    
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, fmax=8000)
    S_dB = librosa.power_to_db(S, ref=np.max)

    _create_spectrogram_figure(S_dB, sr, spec_output_path, x_axis='time', y_axis='mel', fmax=8000)

    # read figure again yuck matplotlib
    image = cv2.imread(spec_output_path)

    local_predictions = data["local_predictions"]
    local_confidences = data["local_confidences"]
    num_predictions = len(local_predictions)

    annotation_width = 640/num_predictions
    annotation_width = math.floor(annotation_width)

    font = cv2.FONT_HERSHEY_SIMPLEX

    for i in range(num_predictions):
        if local_predictions[i] == 1:
            image = cv2.rectangle(image, (i*annotation_width,20), ((i+1)*annotation_width, 460), (255,255,255), 2)
            image = cv2.putText(image, str(local_confidences[i]), (i*annotation_width + 5, 240), font, 0.5, (255, 255, 255), 1, cv2.LINE_AA, False)
            
    image = cv2.putText(image, str(wav_timestamp), (0, 20), font, 0.5, (255, 255, 255), 2, cv2.LINE_AA, False)
    cv2.imwrite(spec_output_path, image)
