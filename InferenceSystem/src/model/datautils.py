import m3u8
import os, sys, glob, json
import random, time, datetime
import uuid
import urllib.request
import shutil
import tqdm
import pdb
import numpy as np
import pandas as pd
from . import params
from pathlib import Path # TODO: make path usage a bit more consistent
from .scraper import download_from_url


def _load_annotation_Json(json_file):
    with open(json_file) as f:
        jtext = f.read()
    return json.loads(jtext.replace('\'','"'))


def _datetime_parser(filename):
    return datetime.datetime.fromtimestamp(int(filename)).strftime('%Y-%m-%d')


def _id_parser(filename):
    return Path(filename).stem.split('_')[0]


def _split_train_dev(tsv, split=0.15):
    wavfiles = tsv["wav_filename"].unique()
    n_wavfiles = len(wavfiles); n_dev_wavfiles = int(n_wavfiles*split)
    print("Splitting {} wavfiles into {} dev files".format(n_wavfiles, n_dev_wavfiles))
    np.random.shuffle(wavfiles)
    dev_wavfiles = np.random.choice(wavfiles, n_dev_wavfiles)
    tsv = tsv.set_index("wav_filename").copy()
    dev_tsv = tsv.loc[dev_wavfiles].reset_index()
    train_tsv = tsv.drop(dev_wavfiles).reset_index()
    return train_tsv, dev_tsv


def make_dataset(annotations_dir, positive_chunks_dir, negative_chunks_dir, 
            dataset_dir, data_source, 
            location=None, date_parser=_datetime_parser, id_parser=_id_parser):
    # annotations, positive wavs, negative wavs, target directory

    json_files = glob.glob(annotations_dir+"/*.json")
    negative_wav_chunks = glob.glob(negative_chunks_dir+"/*.wav")
    tsv_records = []
    wavfile_map = {"positive":[], "negative": []}

    for jsf in json_files:
        js = _load_annotation_Json(jsf)
        wav_chunk_filename = Path(js["uri"]).name
        data_source_id = id_parser(wav_chunk_filename)
        if location is None: location = js["source_guid"]

        for an in js["annotations"]:
            rcd = {
                "wav_filename": wav_chunk_filename,
                "start_time_s": an["start_s"],
                "duration_s": an["duration_s"],
                "location": location,
                "date": date_parser(data_source_id),
                "data_source": data_source,
                "data_source_id": data_source_id 
            }
            tsv_records.append(rcd)

        if len(js["annotations"])>0: wavfile_map["positive"].append(wav_chunk_filename)

    # add chunks containing only negatives 
    for neg_chunk in negative_wav_chunks:
        wav_chunk_filename = Path(neg_chunk).name
        data_source_id = id_parser(wav_chunk_filename)
        if location is None: location = tsv_records[0]["location"]
        rcd = {
            "wav_filename": wav_chunk_filename,
            "start_time_s": 0.000,
            "duration_s": 0.000,
            "location": location, # assume all are from same location 
            "date": date_parser(data_source_id),
            "data_source": data_source,
            "data_source_id": data_source_id
        }
        tsv_records.append(rcd)

        wavfile_map["negative"].append(wav_chunk_filename)
    
    tsv = pd.DataFrame(tsv_records)
    columns = [ "wav_filename", "start_time_s", "duration_s", 
                "location", "date", "data_source", "data_source_id" ]
    tsv = tsv.reindex(columns=columns)
    train_tsv, dev_tsv = _split_train_dev(tsv, split=0.15)

    # copy over files into dataset directory
    print("Copying over files to dataset directory ..")
    wavs_dir = Path(dataset_dir)/"wav"
    os.makedirs(wavs_dir, exist_ok=True)
    for wf in tqdm.tqdm(wavfile_map["positive"]):
        shutil.copy2(Path(positive_chunks_dir)/wf, wavs_dir/wf) 
    for wf in tqdm.tqdm(wavfile_map["negative"]):
        shutil.copy2(Path(negative_chunks_dir)/wf, wavs_dir/wf) 
    
    # write tsv to file
    train_tsv.to_csv(Path(dataset_dir)/"train.tsv", sep='\t', index=False)
    dev_tsv.to_csv(Path(dataset_dir)/"dev.tsv", sep='\t', index=False)

    return tsv, wavfile_map


def compute_dataset_stats(dataset_dir, wav_dataset):
    ## mean, invstd computation, Two-pass
    mean = np.zeros(params.N_MELS)
    for i in tqdm.tqdm(range(len(wav_dataset))):
        mean += wav_dataset[i][0].mean(axis=0) # can use individual means as they're all the same size
    mean /= len(wav_dataset)

    variance = np.zeros(params.N_MELS)
    for i in tqdm.tqdm(range(len(wav_dataset))):
        variance += ((wav_dataset[i][0]-mean)**2).mean(axis=0)
    variance /= len(wav_dataset)
    invstd = 1/np.sqrt(variance)

    # write to file 
    np.savetxt(Path(dataset_dir)/'mean{}.txt'.format(params.N_MELS),mean)
    np.savetxt(Path(dataset_dir)/'invstd{}.txt'.format(params.N_MELS),invstd)


# TODO: Think about how to re-use this for Azure Batch scenario
def download_hls_segment(stream_urls,tmp_root,output_root):
    """
    Downloads ~1 hour HLS file segments
    """
    ffmpeg_cmd = os.environ['ffmpeg']
    assert (ffmpeg_cmd is not None), "Missing ffmpeg: Ensure you have set the ffmpeg env variable."

    for stream_url in stream_urls: 
        # hydrophone node
        hydrophone_node = Path(stream_url).parent.parent.parent.name
        # use tmp_path for intermediate files, clear after 
        segment_posixtime = Path(stream_url).parent.name
        tmp_path = Path(tmp_root)/segment_posixtime
        os.makedirs(tmp_path,exist_ok=True)
        # final file is copied into output_path
        output_path = Path(output_root)/hydrophone_node
        os.makedirs(output_path,exist_ok=True)
        # download all *.ts HLS files to tmp_path
        stream_obj = m3u8.load(stream_url)
        file_names = []
        for audio_segment in stream_obj.segments:
            base_path = audio_segment.base_uri
            file_name = audio_segment.uri
            audio_url = base_path + file_name
            try:
                download_from_url(audio_url,tmp_path)
                file_names.append(file_name)
            except Exception:
                print("Skipping",audio_url,": error.")
        # concatentate all wav files with ffmpeg
        hls_list = "hls_list.txt"
        hls_file = (segment_posixtime+".ts")
        audio_file = (segment_posixtime+".wav")
        with open(tmp_path/hls_list,'w') as f:
            for fn in file_names: f.write("file '{}'\n".format(fn))
        ffmpeg_cli = f'cd {tmp_path} && %ffmpeg% -f concat -i {hls_list} -c copy {hls_file}'
        os.system(ffmpeg_cli)
        # convert to wav file format
        ffmpeg_cli = f'%ffmpeg% -i {tmp_path/hls_file} {output_path/audio_file} -loglevel warning'
        os.system(ffmpeg_cli)
        # clear the tmp_path
        os.system(f'rm -rf {tmp_path}')    


def test_hls_download():
    # These URLs are only two of a longer list.
    make_osl_url = lambda x: "https://s3-us-west-2.amazonaws.com/streaming-orcasound-net/rpi_orcasound_lab/hls/{}/live.m3u8".format(x)

    make_bush_url = lambda x: "https://s3-us-west-2.amazonaws.com/streaming-orcasound-net/rpi_bush_point/hls/{}/live.m3u8".format(x)

    urls = []
    urls.extend([ make_osl_url(u) for u in ["1541140334","1541161938","1543256233","1543613535"] ])
    urls.extend([ make_bush_url(u) for u in ["1539455461","1539498651","1540881062","1545741018"] ])
