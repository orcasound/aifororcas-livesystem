import logging
import urllib.request
import os
import sys
from threading import Thread
import time
from predict_and_aggregate import OrcaDetectionModel
from pathlib import Path
from model.scraper import download_from_url
import m3u8

from datetime import datetime
logger = logging.getLogger(__name__)
import model.params as params
import math
import ffmpeg #ffmpeg-python
import shutil
import spectrogram_visualizer

# TODO: get the list from https://live.orcasound.net/api/json/feeds
ORCASOUND_STREAMS = {
    'BushPoint': 'https://s3-us-west-2.amazonaws.com/audio-orcasound-net/rpi_bush_point',
    'MaSTCenter': 'https://s3-us-west-2.amazonaws.com/audio-orcasound-net/rpi_mast_center',
    'NorthSanJuanChannel': 'https://s3-us-west-2.amazonaws.com/audio-orcasound-net/rpi_north_sjc',
    'OrcasoundLab': 'https://s3-us-west-2.amazonaws.com/audio-orcasound-net/rpi_orcasound_lab',
    'PointRobinson': 'https://s3-us-west-2.amazonaws.com/audio-orcasound-net/rpi_point_robinson',
    'PortTownsend': 'https://s3-us-west-2.amazonaws.com/audio-orcasound-net/rpi_port_townsend',
    'SunsetBay': 'https://s3-us-west-2.amazonaws.com/audio-orcasound-net/rpi_sunset_bay'
}

dirname = os.getcwd()
LIVE_CHUNKED_FILES_PATH = "live_feed2"
live_feed_path = os.path.join(dirname, LIVE_CHUNKED_FILES_PATH, 'Live') # save .ts files for a chunk of audio
spec_live_feed_path = os.path.join(dirname, LIVE_CHUNKED_FILES_PATH, 'LiveSpec') # save spectrogram .png with annotations and 
raw_path = os.path.join(dirname, LIVE_CHUNKED_FILES_PATH, 'Raw') # save .wav file

def read_files_and_chunk_them(args):

    # Create a predictor class
    whalecall_classification_model = OrcaDetectionModel(args.modelPath, args.localPredictionThreshold)

    # make sure the folders exist and clear any older versions of them
    shutil.rmtree(LIVE_CHUNKED_FILES_PATH)
    os.mkdir(LIVE_CHUNKED_FILES_PATH)
    if not os.path.exists(live_feed_path):
        os.mkdir(live_feed_path)
    if not os.path.exists(spec_live_feed_path):
        os.mkdir(spec_live_feed_path)
    if not os.path.exists(raw_path):
        os.mkdir(raw_path)

    counter = 0
    while True:
        for _stream_name, _stream_base in ORCASOUND_STREAMS.items():
            try:
                print("Listening to location {loc}".format(loc=_stream_base))
                latest = f"{_stream_base}/latest.txt"
                stream_id = urllib.request.urlopen(
                    latest).read().decode("utf-8").replace("\n", "")

                stream_url = "{}/hls/{}/live.m3u8".format(
                    (_stream_base), (stream_id))

                counter = download_hls_segment_and_predict(counter, stream_url, _stream_name, args.clipInSeconds, whalecall_classification_model)
                counter = counter % args.spectrogramBufferLength
            except:
                e = sys.exc_info()
                raise Exception("exception is {e}".format(e=e))

        if args.sleepInSeconds > 0:
            print(f"Sleeping for {args.sleepInSeconds} seconds before starting next interation.\n")
            time.sleep(args.sleepInSeconds)

# 
def download_hls_segment_and_predict(counter, stream_url, stream_name, wav_duration, whalecall_classification_model):
    """
    Downloads last wav_duration HLS file segments
    """

    # use tmp_path for intermediate files, clear after
    clipname = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    tmp_root = Path(live_feed_path)/stream_name
    output_root = Path(raw_path)/stream_name

    tmp_path = Path(tmp_root)/clipname
    os.makedirs(tmp_path,exist_ok=True)
    os.makedirs(output_root,exist_ok=True)
    
    # download all *.ts HLS files to tmp_path
    stream_obj = m3u8.load(stream_url)

    num_total_segments = len(stream_obj.segments)
    num_segments_in_wav_duration = math.ceil(wav_duration/stream_obj.target_duration)

    if num_total_segments - num_segments_in_wav_duration < 0:
        return counter
    
    segment_start_index = max(num_total_segments - num_segments_in_wav_duration, 0)

    file_names = []
    for i in range(segment_start_index, num_total_segments):
        audio_segment = stream_obj.segments[i]
        base_path = audio_segment.base_uri
        file_name = audio_segment.uri
        audio_url = base_path + file_name
        try:
            download_from_url(audio_url,tmp_path)
            file_names.append(file_name)
        except Exception:
            print("Skipping",audio_url,": error.")

    # concatentate all .ts files with ffmpeg
    hls_file = (clipname+".ts")
    audio_file = (clipname+".wav")
    filenames_str = " ".join(file_names)
    concat_ts_cmd = "cd {tp} && cat {fstr} > {hls_file}".format(tp=tmp_path, fstr=filenames_str, hls_file=hls_file)
    os.system(concat_ts_cmd)
    
    # read the concatenated .ts and write to wav
    stream = ffmpeg.input(os.path.join(tmp_path,Path(hls_file)))
    stream = ffmpeg.output(stream, os.path.join(output_root, audio_file))
    ffmpeg.run(stream, quiet=True)

    # clear the tmp_path
    os.system(f'rm -rf {tmp_path}')

    # make a prediction - writes to Azure DB
    wav_file_path = os.path.join(output_root, audio_file)
    result_json = whalecall_classification_model.predict_and_aggregate(wav_file_path)

    # writes a .png with annotations in the spec_live directory, audio is also copied to spec_live
    stream_spec_path = Path(spec_live_feed_path)/stream_name
    os.makedirs(stream_spec_path, exist_ok=True)

    spectrogram_output_prefix = os.path.join(stream_spec_path, str(counter))
    wav_player_path = spectrogram_output_prefix + ".wav"
    spec_player_path = spectrogram_output_prefix + ".png"
    shutil.copy(wav_file_path, wav_player_path)

    spectrogram_visualizer.write_annotations_on_spectrogram(wav_player_path, clipname, result_json, spec_player_path)
    return counter+1
