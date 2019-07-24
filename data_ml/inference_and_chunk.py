import os, json
import torch
import numpy as np
import src.params as params
import pdb

from src.model import get_model_or_checkpoint
from scipy.io import wavfile
from collections import defaultdict
from src.dataloader import AudioFileWindower 
from pathlib import Path
# inputs: model, mean invstd, input_wav, output_chunk_dir, output_json_dir

# iterate through windows and save audio chunks
def inference_and_write_chunks(
    audio_file_windower,output_chunk_dir,predictions_dir,model_path,
    chunk_duration=params.INFERENCE_CHUNK_S):

    # initialize model from checkpoint
    model, _ = get_model_or_checkpoint(params.MODEL_NAME,model_path,use_cuda=True)
    blob_root = "https://podcaststorage.blob.core.windows.net/dummydata/wavmaster_chunks"

    # iterate through windows in dataloader, store current chunk windows and length
    curr_chunk, curr_chunk_duration, curr_chunk_json = [], 0, {}
    file_chunk_counts = defaultdict(int)

    for i in range(len(audio_file_windower)):

        # first get an audio window 
        audio_file_windower.get_mode = 'audio'
        audio_window, _ = audio_file_windower[i]
        _, _, _, af = audio_file_windower.windows[i]
        window_s = audio_file_windower.window_s
        # details for the current chunk
        postfix = format(file_chunk_counts[af.name],'04x')
        chunk_file_name = (Path(af.name).stem+postfix+'.wav')
        output_file_path = output_chunk_dir / chunk_file_name 
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
        confidence = np.exp(pred.detach().cpu().numpy())
        pred_id = torch.argmax(pred, dim=1).item()

        # trigger and write JSON if positive prediction 
        if pred_id==1: # #TODO: maybe confidence-based filtering?
            if len(curr_chunk_json)==0:
                # add the header fields (uri, absolute_time, source_guid, annotations)
                curr_chunk_json["uri"] = blob_uri
                curr_chunk_json["absolute_time"] = 1543804333 # fake
                curr_chunk_json["source_guid"] = "WHOIS" 
                curr_chunk_json["annotations"] = [] 
            start_s, duration_s = curr_chunk_duration-window_s, window_s 
            curr_chunk_json["annotations"].append({"start_time_s":start_s,"duration_s":duration_s})
            print("Positive prediction at",start_s,"!")

        # if exceeds chunk_duration, write chunk and JSON and reset
        if curr_chunk_duration > chunk_duration:
            # write out the chunk and reset
            print("Writing out chunk:",output_file_path.name)
            wavfile.write(output_file_path,af.sr,np.concatenate(curr_chunk))
            # write JSON to file and clear
            with open(predictions_dir/(Path(chunk_file_name).stem+".json"),'w') as fp:
                json.dump(curr_chunk_json,fp)
            # clearing up this chunk
            curr_chunk_json = {}
            curr_chunk, curr_chunk_duration = [], 0.
            file_chunk_counts[af.name] += 1


if __name__ == "__main__":
    wavmaster_path = Path("../data/wavmaster")
    output_chunk_dir = Path("../data/wavmaster_chunked")
    preds_dir = Path("../data/wavmaster_chunked_preds")
    model_path = Path("../models/AudioSet_fc_all")
    mean, invstd = model_path/params.MEAN_FILE, model_path/params.INVSTD_FILE 
    wav_file_paths = [ wavmaster_path/p for p in os.listdir(wavmaster_path) ]
    windower = AudioFileWindower(wav_file_paths,mean=mean,invstd=invstd)
    inference_and_write_chunks(windower,output_chunk_dir,preds_dir,model_path)
