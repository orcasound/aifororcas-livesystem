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

def write_annotations_on_spectrogram(wav_file_path, wav_timestamp, data, spec_output_path):
    """

    """

    y, sr = librosa.load(wav_file_path)
    D = np.abs(librosa.stft(y))**2
    
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, fmax=8000)
    S_dB = librosa.power_to_db(S, ref=np.max)

    pylab.axis('off') # no axis
    pylab.axes([0., 0., 1., 1.], frameon=False, xticks=[], yticks=[]) # Remove the white edge
    librosa.display.specshow(S_dB, x_axis='time',y_axis='mel', sr=sr, fmax=8000)
    pylab.savefig(spec_output_path, bbox_inches=None, pad_inches=0)

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
