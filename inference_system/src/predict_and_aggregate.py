import os, sys, json, glob
import torch
import numpy as np
import model.params as params
import argparse, pdb

from model.model import get_model_or_checkpoint
from scipy.io import wavfile
from collections import defaultdict
from model.dataloader import AudioFileWindower 
from pathlib import Path

import AzureStorage
from datetime import datetime
import numpy as np
import spectrogram_visualizer

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
    def __init__(self, model_path, threshold=0.5, global_aggregation_percentile_threshold=80):
        #i initialize model
        self.model, _ = get_model_or_checkpoint(params.MODEL_NAME,model_path,use_cuda=False)
        self.mean = os.path.join(model_path, params.MEAN_FILE)
        self.invstd = os.path.join(model_path, params.INVSTD_FILE)
        self.threshold = threshold
        self.global_aggregation_percentile_threshold = global_aggregation_percentile_threshold

    def split_and_predict(self, wav_file_path):
        """
        Args contains:
            - wavfile_path
            - model_path 
        """

        # initialize parameters
        wavfile_path = wav_file_path
        chunk_duration=params.INFERENCE_CHUNK_S

        audio_file_windower = AudioFileWindower([wavfile_path], mean=self.mean, invstd=self.invstd)
        window_s = audio_file_windower.window_s

        # initialize output JSON
        result_json = {
            "local_predictions":[],
            "local_confidences":[]
            }

        # iterate through dataloader and add accumulate predictions
        for i in range(len(audio_file_windower)):
            # get a mel spec for the window 
            audio_file_windower.get_mode = 'mel_spec'
            mel_spec_window, _ = audio_file_windower[i]
            # run inference on window
            input_data = torch.from_numpy(mel_spec_window).float().unsqueeze(0).unsqueeze(0)
            pred, embed = self.model(input_data)
            posterior = np.exp(pred.detach().cpu().numpy())
            pred_id = 1 if posterior[0,1] > self.threshold else 0;
            # pred_id = torch.argmax(pred, dim=1).item()
            #TODO@Akash: correct these pred_ids to be created with a threshold.
            # current argmax implies threshold of 0.5
            confidence = round(float(posterior[0,1]),3)

            result_json["local_predictions"].append(pred_id)
            result_json["local_confidences"].append(confidence)

        return result_json


    def aggregate_predictions(self, result_json):
        """
        Given N local window predictions Pi, aggregate into a global one.
        Current logic is very scrappy. 

        Global prediction = avg(Pi) > threshold
        """

        # calculate nth percentile of result_json["local_confidences"], this is global confidence
        local_confidences = result_json["local_confidences"]
        global_percentile = np.percentile(local_confidences, self.global_aggregation_percentile_threshold)
        result_json["global_confidence"] = global_percentile
        
        return result_json

    def predict_and_aggregate(self, wav_file_path):
        result_json = self.split_and_predict(wav_file_path)
        result_json = self.aggregate_predictions(result_json)

        clipname = os.path.basename(wav_file_path)
        clipname_without_ext = clipname.split(".wav")[0]
        tokens = clipname_without_ext.split('-')
        partitionkey = "-".join(tokens[0:3])
        rowkey = "-".join(tokens[3:])

        annotations = result_json.copy()

        # why does dump to db modify the object?
        AzureStorage.dump2db(result_json, partitionkey, rowkey)
        return annotations

    def predict(self, wav_file_path):
        result_json = self.split_and_predict(wav_file_path)
        result_json = self.aggregate_predictions(result_json)
        return result_json