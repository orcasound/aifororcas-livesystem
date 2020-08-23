import os, sys, json, glob
import torch
import numpy as np
import src.params as params
import argparse, pdb

from src.model import get_model_or_checkpoint
from scipy.io import wavfile
from collections import defaultdict
from src.dataloader import AudioFileWindower 
from pathlib import Path

import AzureStorage
from datetime import datetime

import spectrogram_visualizer
import pdb

WAV_SR = 44100
# WAV_SR = params.SAMPLE_RATE 


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
    def __init__(self, model_path, threshold=0.5):
        #i initialize model
        self.model, _ = get_model_or_checkpoint(params.MODEL_NAME,model_path,use_cuda=False)
        self.mean = os.path.join(model_path, params.MEAN_FILE)
        self.invstd = os.path.join(model_path, params.INVSTD_FILE)
        self.threshold = threshold

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

        result_json["global_confidence"] = [np.mean(result_json["local_confidences"])]
        result_json["global_prediction"] = 1 if result_json["global_confidence"][0] > 0.5 else 0

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


# if __name__ == "__main__":
#     """
#     Processes unlabelled data using a classifier with two operating points/thresholds.
#     1. Generating positive annotation candidates for Pod.Cast UI. Use a threshold here that favors high recall (>85% ish) of positive examples over precision (>65% ish). 
#     2. Generating negative examples from this distribution. Use a threshold here that's high precision (>90%) as we don't want positive examples incorrectly labelled negative. 

#     These values above are approximate and will likely evolve as the classifier keeps improving. 
#     NOTE: The wavfile names are assumed to be the "absolute_time" below. 

#     Outputs:
#     1. For positive candidates: corresponding 60s chunks of wavfiles and JSON with schema:
#     {
#         "uri": "https://podcaststorage.blob.core.windows.net/[RELATIVE BLOB PATH]/[WAVCHUNK NAME],
#                 e.g. https://podcaststorage.blob.core.windows.net/orcasoundlabchunked/1562337136_000f.wav
#         "absolute_time": UNIX time of corresponding Orcasound S3 bucket e.g. 1562337136,
#         "source_guid": Orcasound lab hydrophone id e.g. rpi_orcasound_lab, 
#         "annotations": [
#             {"start_time_s","duration_s","confidence"}
#         ]
#     }
#     2. For negative examples: corresponding 60s chunks of wavfiles that are labelled as all negative with high confidence. 
#     """
#     parser = argparse.ArgumentParser()
#     parser.add_argument('-wavfilePath', default=None, type=str, required=True)
#     parser.add_argument('-modelPath', default='AudioSet_fc_all', type=str, required=True)
#     parser.add_argument('-outputPath', type=str, required=True)
#     args = parser.parse_args()

#     test_predict_and_aggregate(args)
#     # inference_and_write_chunks(args)
