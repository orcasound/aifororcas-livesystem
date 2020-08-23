# HLSStream class 
import urllib.request
import os
import m3u8
import math
import ffmpeg #ffmpeg-python
from model.scraper import download_from_url
from datetime import datetime
import time
from pathlib import Path

#TODO (@prgogia) Handle errors due to rebooting of hydrophone
class HLSStream():
    """
    stream_base = 'https://s3-us-west-2.amazonaws.com/streaming-orcasound-net/rpi_orcasound_lab'
    polling_interval = 60 sec
    """

    def __init__(self, stream_base, polling_interval, wav_dir):
        self.stream_base = stream_base
        self.polling_interval = polling_interval
        self.wav_dir = wav_dir

    # this function grabs audio from last_end_time to 
    def get_next_clip(self, current_clip_end_time):
        
        #TODO(@prgogia) fix any tiny errors here
        # if current time < current_clip_end_time, sleep for the difference
        now = datetime.utcnow()

        # the extra 10 seconds to sleep is to download the last .ts segment properly
        time_to_sleep = (current_clip_end_time-now).total_seconds() + 10

        if time_to_sleep < 0:
            print("Issue with timing")

        if time_to_sleep > 0:
            time.sleep(time_to_sleep)

        #TODO(@prgogia) this should be clip-start time
        clipname = current_clip_end_time.isoformat() + "Z"

        # get latest AWS bucket
        print("Listening to location {loc}".format(loc=self.stream_base))
        latest = f"{self.stream_base}/latest.txt"
        stream_id = urllib.request.urlopen(
            latest).read().decode("utf-8").replace("\n", "")

        # stream_url for the current AWS bucket
        stream_url = "{}/hls/{}/live.m3u8".format(
            (self.stream_base), (stream_id))

        # Create tmp path to hold .ts segments
        tmp_path = "tmp_path"
        os.makedirs(tmp_path,exist_ok=True)

        stream_obj = m3u8.load(stream_url)
        num_total_segments = len(stream_obj.segments)
        num_segments_in_wav_duration = math.ceil(self.polling_interval/stream_obj.target_duration)

        if num_total_segments < num_segments_in_wav_duration:
            return None

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
        wav_file_path = os.path.join(self.wav_dir, audio_file)
        filenames_str = " ".join(file_names)
        concat_ts_cmd = "cd {tp} && cat {fstr} > {hls_file}".format(tp=tmp_path, fstr=filenames_str, hls_file=hls_file)
        os.system(concat_ts_cmd)
        
        # read the concatenated .ts and write to wav
        stream = ffmpeg.input(os.path.join(tmp_path, Path(hls_file)))
        stream = ffmpeg.output(stream, wav_file_path)
        ffmpeg.run(stream, quiet=True)

        # clear the tmp_path
        os.system(f'rm -rf {tmp_path}')

        return wav_file_path, clipname

