import os, json, glob
import torch
import numpy as np
import src.params as params
import argparse

from src.model import get_model_or_checkpoint
from scipy.io import wavfile
from collections import defaultdict
from src.dataloader import AudioFileWindower 
from pathlib import Path


# iterate through windows and save audio chunks and prediction candidates
def inference_and_write_chunks(args):

    # load dataset object used to iterate windows of audio
    chunk_duration=params.INFERENCE_CHUNK_S
    wav_file_paths = [ Path(p) for p in glob.glob(args.wavMasterPath+"/*.wav") ]
    model_path = Path(args.modelPath)
    mean, invstd = model_path/params.MEAN_FILE, model_path/params.INVSTD_FILE 
    audio_file_windower = AudioFileWindower(wav_file_paths,mean=mean,invstd=invstd)

    # initialize model from checkpoint
    model, _ = get_model_or_checkpoint(params.MODEL_NAME,model_path,use_cuda=True)

    # various output locations
    blob_root = "https://podcaststorage.blob.core.windows.net/{}".format(args.relativeBlobPath)
    pos_chunk_dir = Path(args.positiveChunkDir)
    pos_preds_dir = Path(args.positiveCandidatePredsDir)
    neg_chunk_dir = Path(args.negativeChunkDir)
    os.makedirs(pos_chunk_dir,exist_ok=True)
    os.makedirs(pos_preds_dir,exist_ok=True)
    os.makedirs(neg_chunk_dir,exist_ok=True)

    # iterate through windows in dataloader, store current chunk windows and length
    curr_chunk, curr_chunk_json = [], {}
    curr_chunk_duration, curr_chunk_all_negative = 0, 1
    file_chunk_counts = defaultdict(int)

    for i in range(len(audio_file_windower)):

        # first get an audio window 
        audio_file_windower.get_mode = 'audio'
        audio_window, _ = audio_file_windower[i]
        _, _, _, af = audio_file_windower.windows[i]
        window_s = audio_file_windower.window_s
        # details for the current chunk
        postfix = '_'+format(file_chunk_counts[af.name],'04x')
        chunk_file_name = (Path(af.name).stem+postfix+'.wav')
        absolute_time = Path(af.name).stem  # NOTE: assumes filename is the absolute time 
        pos_chunk_path = pos_chunk_dir / chunk_file_name 
        neg_chunk_path = neg_chunk_dir / chunk_file_name 
        blob_uri = blob_root+'/'+chunk_file_name

        # add window to current chunk
        curr_chunk_duration += window_s
        curr_chunk.append(audio_window)

        # get a mel spec for the window 
        audio_file_windower.get_mode = 'mel_spec'
        mel_spec_window, _ = audio_file_windower[i]
        # run inference on window
        input_data = torch.from_numpy(mel_spec_window).float().unsqueeze(0).unsqueeze(0)
        pred, embed = model(input_data)
        posterior = np.exp(pred.detach().cpu().numpy())
        pred_id = torch.argmax(pred, dim=1).item()
        confidence = round(float(posterior[0,1]),3)

        # trigger and update JSON for current chunk if positive prediction 
        # chunk is considered negative if there are no positive candidates and 
        # all windows had confidence < negativeThreshold
        if confidence>args.positiveThreshold:  
            if len(curr_chunk_json)==0:
                # add the header fields (uri, absolute_time, source_guid, annotations)
                curr_chunk_json["uri"] = blob_uri
                curr_chunk_json["absolute_time"] = absolute_time 
                curr_chunk_json["source_guid"] = "rpi_orcasound_lab" 
                curr_chunk_json["annotations"] = [] 
            start_s, duration_s = curr_chunk_duration-window_s, window_s 
            curr_chunk_json["annotations"].append(
                    {
                        "start_time_s":start_s,
                        "duration_s":duration_s,
                        "confidence":confidence
                    }
                )
            print("Positive prediction at {:.2f}, Confidence {:.3f}!".format(start_s,confidence))
            curr_chunk_all_negative *= 0
        elif confidence>args.negativeThreshold:
            curr_chunk_all_negative *= 0

        # if exceeds chunk_duration, write chunk and JSON and reset
        if curr_chunk_duration > chunk_duration:

            # if there are predictions, write JSON and positive chunk to file and clear
            if len(curr_chunk_json)!=0:
                with open(pos_preds_dir/(Path(chunk_file_name).stem+".json"),'w') as fp:
                    json.dump(curr_chunk_json,fp)
                curr_chunk_json = {}
                print("Writing out positive candidate chunk:",pos_chunk_path.name)
                wavfile.write(pos_chunk_path,af.sr,np.concatenate(curr_chunk))
            elif curr_chunk_all_negative: 
                print("Writing out negative chunk:",neg_chunk_path.name)
                wavfile.write(neg_chunk_path,af.sr,np.concatenate(curr_chunk))

            # clearing up this chunk
            curr_chunk, curr_chunk_duration = [], 0.
            curr_chunk_all_negative = 1
            file_chunk_counts[af.name] += 1


if __name__ == "__main__":
    """
    Processes unlabelled data using a classifier with two operating points/thresholds.
    1. Generating positive annotation candidates for Pod.Cast UI. Use a threshold here that favors high recall (>85% ish) of positive examples over precision (>65% ish). 
    2. Generating negative examples from this distribution. Use a threshold here that's high precision (>90%) as we don't want positive examples incorrectly labelled negative. 

    These values above are approximate and will likely evolve as the classifier keeps improving. 
    NOTE: The wavfile names are assumed to be the "absolute_time" below. 

    Outputs:
    1. For positive candidates: corresponding 60s chunks of wavfiles and JSON with schema:
    {
        "uri": "https://podcaststorage.blob.core.windows.net/[RELATIVE BLOB PATH]/[WAVCHUNK NAME],
                e.g. https://podcaststorage.blob.core.windows.net/orcasoundlabchunked/1562337136_000f.wav
        "absolute_time": UNIX time of corresponding Orcasound S3 bucket e.g. 1562337136,
        "source_guid": Orcasound lab hydrophone id e.g. rpi_orcasound_lab, 
        "annotations": [
            {"start_time_s","duration_s","confidence"}
        ]
    }
    2. For negative examples: corresponding 60s chunks of wavfiles that are labelled as all negative with high confidence. 
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('-wavMasterPath', default=None, type=str, required=True)
    parser.add_argument('-sourceGuid', default=None, type=str, required=True)
    parser.add_argument('-modelPath', default='AudioSet_fc_all', type=str, required=True)
    parser.add_argument('-positiveChunkDir', default=None, type=str, required=True)
    parser.add_argument('-positiveCandidatePredsDir', default=None, type=str, required=True)
    parser.add_argument('-positiveThreshold', default=None, type=float, required=True)
    parser.add_argument('-relativeBlobPath', default=None, type=str, required=True)
    parser.add_argument('-negativeChunkDir', default=None, type=str, required=True)
    parser.add_argument('-negativeThreshold', default=None, type=float, required=True)

    args = parser.parse_args()
    inference_and_write_chunks(args)
