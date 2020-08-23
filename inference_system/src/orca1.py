import logging
import urllib.request
import os
import sys
from threading import Thread
import time
import numpy as np
import pandas as pd
import wave
import pylab
import gc
import glob
import shutil

import json
import torch.nn as nn
import torch

import logging
import random

import librosa 
import librosa.display


logger = logging.getLogger(__name__)


ORCASOUND_STREAMS = {
    'OrcasoundLab': 'https://s3-us-west-2.amazonaws.com/streaming-orcasound-net/rpi_orcasound_lab',
    'BushPoint': 'https://s3-us-west-2.amazonaws.com/streaming-orcasound-net/rpi_bush_point',
    #'PortTownsend': 'https://s3-us-west-2.amazonaws.com/streaming-orcasound-net/rpi_port_townsend'
}

LIVE_FEED_PATH = None

LIVE_CHUNKED_FILES_PATH = "live_feed2"

dirname = os.getcwd()
live_feed_path = os.path.join(dirname, LIVE_CHUNKED_FILES_PATH, 'Live')
spec_live_feed_path = os.path.join(dirname, LIVE_CHUNKED_FILES_PATH, 'LiveSpec')
raw_path = os.path.join(dirname, LIVE_CHUNKED_FILES_PATH, 'Raw')

def initModel(path):
    labels = ['neg','pos']
    empty_data = ImageDataBunch.single_from_classes(
        path, labels, ds_tfms =get_transforms(), size=224).normalize(imagenet_stats)
    learn = create_cnn(empty_data, models.resnet34)
    learn = learn.load('model')
    return learn

def predict(imgfile):
    img= open_image(imgfile)
    pred_class, pred_idx, outputs = learn.predict(img)        
    return str(pred_class)

def getsoundspregram(audio_file, spec_file):
    samples, sample_rate = librosa.load(audio_file)
    fig = plt.figure(figsize=[0.72,0.72])
    ax = fig.add_subplot(111)
    ax.axes.get_xaxis().set_visible(False)
    ax.axes.get_yaxis().set_visible(False)
    ax.set_frame_on(False)
    filename  = spec_file
    S = librosa.feature.melspectrogram(y=samples, sr=sample_rate)
    librosa.display.specshow(librosa.power_to_db(S, ref=np.max))
    plt.savefig(spec_file, dpi=400, bbox_inches='tight',pad_inches=0)
    plt.close('all')
    return filename

def generate_spectrogram():
    '''
    Helper function generate spectrogram
    '''
    wav_files = [f for f in glob.glob(live_feed_path + "**/*.wav", recursive=True)]
    for wav_file in wav_files:  
        local_wav_file_path = wav_file
        spectogram_file_path = os.path.join(spec_live_feed_path, os.path.basename(local_wav_file_path).replace('.wav','.png'))
        getsoundspregram(local_wav_file_path, spectogram_file_path)

def read_files_and_chunk_them(sleep_seconds):
    counter = 0
    # while counter < 3:
    while True:

        counter = counter + 1
        logger.debug("Running iteration {iteration}".format(iteration=counter))

        threads = []
        for _stream_name, _stream_base in ORCASOUND_STREAMS.items():
            try:

                logger.debug("Listening to location {loc}".format(loc=_stream_base))
                # get the ID of the latest stream and build URL to load
                latest = f"{_stream_base}/latest.txt"
                stream_id = urllib.request.urlopen(
                    latest).read().decode("utf-8").replace("\n", "")
                stream_url = "{}/hls/{}/live.m3u8".format(
                    (_stream_base), (stream_id))

                chunked_samples_path = live_feed_path
                logger.debug("Chunked files would be stored in {path}".format(path=chunked_samples_path))

                # make sure the folders exist
                if not os.path.exists(LIVE_CHUNKED_FILES_PATH):
                    os.mkdir(LIVE_CHUNKED_FILES_PATH)
                if not os.path.exists(chunked_samples_path):
                    os.mkdir(chunked_samples_path)
                if not os.path.exists(spec_live_feed_path):
                    os.mkdir(spec_live_feed_path)
                if not os.path.exists(raw_path):
                    os.mkdir(raw_path)

                thread = Thread(target=save_audio_segments, args=(stream_url,
                                                                  _stream_name,
                                                                  1,
                                                                  1,
                                                                  None,
                                                                  chunked_samples_path,
                                                                  False))
                threads.append(thread)
                thread.start()
            except:
                e = sys.exc_info()
                logger.debug("exception is {e}".format(e=e))
                print(f'Unable to load stream from {stream_url}')

        generate_spectrogram()

        for _stream_name, _stream_base in ORCASOUND_STREAMS.items():
            #  Example raw_stream_path :
            #     OrcaStream\live_feed2\Raw\OrcasoundLab
            #     OrcaStream\live_feed2\Raw\BushPoint
            raw_stream_path = os.path.join(raw_path, _stream_name)

            wav_stream_filename = os.path.join(live_feed_path, f"{_stream_name}_00.wav")
            png_stream_filename = os.path.join(spec_live_feed_path, f"{_stream_name}_00.png")

            if not os.path.exists(raw_stream_path):
                os.mkdir(raw_stream_path)

            if os.path.exists(wav_stream_filename) and os.path.exists(png_stream_filename):
                wav_stream_filename = os.path.join(live_feed_path, f"{_stream_name}_00.wav")
                wav_output_filename = os.path.join(raw_stream_path, f"{_stream_name}_{counter}.wav")
                shutil.move(wav_stream_filename,wav_output_filename)

                png_stream_filename = os.path.join(spec_live_feed_path, f"{_stream_name}_00.png")
                png_output_filename = os.path.join(raw_stream_path, f"{_stream_name}_{counter}.png")
                shutil.move(png_stream_filename,png_output_filename)

                results = predict(png_output_filename)
                print(f'Prediction : {png_output_filename} : {results}')

        if sleep_seconds > 0:
            print(f"Sleeping for {sleep_seconds} seconds before starting next interation.\n")
            time.sleep(sleep_seconds)


def save_audio_segments(stream_url,
                        stream_name,
                        segment_seconds,
                        iteration_seconds,
                        mix_with,
                        output_path,
                        verbose):
    """
    Uses ffmpeg (via CLI) to retrieve audio segments from audio_url
    and saves it to the local output_path.

    If mix_with exists, it mixes this file with the live feed source.
    The result
    """

    file_name = f"{stream_name}_%02d.wav"
    output_file = os.path.join(output_path, file_name)


    mix_with_command = ''
    # if os.path.exists(mix_with):
    #     print(f'Mixing with {mix_with}')
    #     mix_with_command = f'-i {mix_with} -filter_complex amix=inputs=2:duration=first'

    ffmpeg_cli = f'ffmpeg -y -i {stream_url} {mix_with_command} -t {iteration_seconds} -f segment -segment_time {segment_seconds} {output_file}'

    print('ffmpeg_cli : {0}'.format(ffmpeg_cli))


    if not verbose:
        ffmpeg_cli = ffmpeg_cli + ' -loglevel error'
    os.system(ffmpeg_cli)

def get_wav_info(wav_file):
    '''
    Extracts information about the wav file to be used for creating the spectograms
    '''
    wav = wave.open(wav_file, 'r')
    frames = wav.readframes(-1)
    sound_info = pylab.frombuffer(frames, 'int16')
    frame_rate = wav.getframerate()
    wav.close()
    return sound_info, frame_rate

def graph_spectrogram(wav_file, serialnumber, audio_begin_TimeStamp, start_second, output_file_path):
    '''
    Creates and saves the spectogram
    '''
    sound_info, frame_rate = get_wav_info(wav_file)
    
    plt.figure(num=None, figsize=(19, 12))
    plt.subplot(222)
    ax = plt.axes()
    ax.set_axis_off()
    plt.specgram(sound_info, Fs=frame_rate)
    
    serialnumber = str(serialnumber)
    audio_begin_TimeStamp = str(audio_begin_TimeStamp)
    plt.savefig(output_file_path, bbox_inches='tight', transparent=True, pad_inches=0.0)
    plt.close()
    gc.collect()  # clear-up memory


learn = initModel(dirname)

listen_thread = Thread(target=read_files_and_chunk_them, args=(4,))
listen_thread.daemon = True
listen_thread.start()


if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    # logging.basicConfig(filename='example.log', filemode='w', level=logging.DEBUG)
    # read_files_and_chunk_them(2)
    while True:
        selection = input("Press Q to quit\n")
        if selection == "Q" or selection == "q":
            print("Quitting...")
            break
