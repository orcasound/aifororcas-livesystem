import argparse
import gc
import glob
import json
import os
import pdb
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import model.params as params
import numpy as np
import pandas as pd
import torch
from model.dataloader import AudioFileWindower
from model.model import get_model_or_checkpoint
from scipy.io import wavfile
from tqdm import tqdm

"""
Input: wav file 
Output: prediction

Steps:
    * Dataloader splits things into windows 
    * Iterate and accumulate predictions 
    - Call aggregation function 
    - Write out a JSON file 

"""
def write_json(result_json, output_path):
    with open(output_path, 'w') as f:
        json.dump(result_json, f)

class OrcaDetectionModel():
    def __init__(self, model_path, threshold=0.7, min_num_positive_calls_threshold=3, hop_s=2.45, rolling_avg=False):
        #i initialize model
        self.model, _ = get_model_or_checkpoint(params.MODEL_NAME,model_path,use_cuda=False)
        self.model.eval()
        self.mean = os.path.join(model_path, params.MEAN_FILE)
        self.invstd = os.path.join(model_path, params.INVSTD_FILE)
        self.threshold = threshold
        self.min_num_positive_calls_threshold = min_num_positive_calls_threshold
        self.hop_s = hop_s
        self.rolling_avg = rolling_avg

    def split_and_predict(self, wav_file_path):
        """
        Args contains:
            - wavfile_path
            - model_path 
        """

        # initialize parameters
        wavfile_path = wav_file_path
        chunk_duration=params.INFERENCE_CHUNK_S

        audio_file_windower = AudioFileWindower(
                [wavfile_path], mean=self.mean, invstd=self.invstd, hop_s=self.hop_s
            )
        window_s = audio_file_windower.window_s

        # initialize output JSON
        result_json = {
            "local_predictions":[],
            "local_confidences":[]
            }

        # iterate through dataloader and add accumulate predictions
        num_windows = len(audio_file_windower)
        for i in tqdm(range(num_windows)):
            # get a mel spec for the window 
            audio_file_windower.get_mode = 'mel_spec'
            mel_spec_window, _ = audio_file_windower[i]
            
            # run inference on window
            with torch.no_grad():  # Disable gradient computation to save memory
                input_data = torch.from_numpy(mel_spec_window).float().unsqueeze(0).unsqueeze(0)
                pred, _ = self.model(input_data)
                posterior = np.exp(pred.detach().cpu().numpy())

            pred_id = 0
            if posterior[0,1] > self.threshold:
                pred_id = 1
            confidence = round(float(posterior[0,1]),3)

            result_json["local_predictions"].append(pred_id)
            result_json["local_confidences"].append(confidence)
            
            # Cleanup tensors every 10 windows to prevent accumulation
            if i > 0 and i % 10 == 0:
                del input_data, pred, posterior, mel_spec_window
                gc.collect()
        
        # Final cleanup before DataFrame creation
        del audio_file_windower
        gc.collect()
        
        submission = pd.DataFrame(dict(
            wav_filename=Path(wav_file_path).name,
            start_time_s=[i*self.hop_s for i in range(num_windows)],
            duration_s=self.hop_s,
            confidence=result_json['local_confidences']
        ))
        if self.rolling_avg:
            rolling_scores = submission['confidence'].rolling(2).mean()
            rolling_scores[0] = submission['confidence'][0]
            submission['confidence'] = rolling_scores
            result_json["local_confidences"] = submission['confidence'].tolist()
        result_json['submission'] = submission

        return result_json


    def aggregate_predictions(self, result_json):
        """
        Given N local window predictions Pi, aggregate into a global one.
        Currently we try to reduce false positives so have strict thresholds
        """

        # calculate nth percentile of result_json["local_confidences"], this is global confidence
        local_confidences = result_json["local_confidences"]
        local_predictions = result_json["local_predictions"]

        pred_array = np.array(local_predictions)
        conf_array = np.array(local_confidences)
        total_num_positive_predictions = sum(pred_array)

        global_prediction = 0
        if total_num_positive_predictions >= self.min_num_positive_calls_threshold:
            global_prediction = 1
        result_json["global_prediction"] = global_prediction
        
        positive_predictions_conf = conf_array[pred_array == 1]
        global_confidence = 0
        if positive_predictions_conf.size > 0:
            global_confidence = np.average(positive_predictions_conf)
        result_json["global_confidence"] = global_confidence*100

        return result_json

    def predict(self, wav_file_path):
        result_json = self.split_and_predict(wav_file_path)
        result_json = self.aggregate_predictions(result_json)
        return result_json
