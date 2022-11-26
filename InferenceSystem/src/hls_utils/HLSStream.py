# HLSStream class 
import urllib.request
import os
import m3u8
import math
import ffmpeg #ffmpeg-python
from . import scraper
from datetime import datetime
from datetime import timedelta

import time
from pytz import timezone
from . import datetime_utils
from pathlib import Path

import boto3
from botocore import UNSIGNED
from botocore.config import Config

def get_readable_clipname(hydrophone_id, cliptime_utc):
    # cliptime is of the form 2020-09-27T00/16/55.677242Z
    cliptime_utc = timezone('UTC').localize(cliptime_utc)
    date = cliptime_utc.astimezone(timezone('US/Pacific'))
    date_format='%Y_%m_%d_%H_%M_%S_%Z'
    clipname = date.strftime(date_format)
    return hydrophone_id + "_" + clipname, date

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
        bucket_folder = self.stream_base.split("https://s3-us-west-2.amazonaws.com/")[1]
        tokens = bucket_folder.split("/")
        self.s3_bucket = tokens[0]
        self.hydrophone_id = tokens[1]

    # this function grabs audio from last_end_time to 
    def get_next_clip(self, current_clip_end_time):

        print("Current clip end time at start " + current_clip_end_time)

        # if current time < current_clip_end_time, sleep for the difference
        now = datetime.utcnow()

        # the extra 10 seconds to sleep is to download the last .ts segment properly
        time_to_sleep = (current_clip_end_time-now).total_seconds() + 10

        if time_to_sleep < 10:
            # This implies that we took more than polling interval seconds to do our processing
            # Ideally, we should set current_clip_end_time = now at this point
            current_clip_end_time = now
            time.sleep(10)

            print("Fell behind now so fast forwarding current_clip_end_time")
        else:
            # In steady state, time_to_sleep is ~41 seconds
            # the very first time, it's ~20 seconds
            time.sleep(time_to_sleep)

        print("Current clip end time after sleep " + current_clip_end_time)
        current_clip_start_time = current_clip_end_time - timedelta(0,60)
        clip_start_time = current_clip_start_time.isoformat() + "Z"
        clipname, _ = get_readable_clipname(self.hydrophone_id, current_clip_start_time)

        # get latest AWS bucket
        print("Listening to location {loc}".format(loc=self.stream_base))
        latest = f"{self.stream_base}/latest.txt"
        stream_id = urllib.request.urlopen(
            latest).read().decode("utf-8").replace("\n", "")

        # stream_url for the current AWS bucket
        stream_url = "{}/hls/{}/live.m3u8".format(
            (self.stream_base), (stream_id))
        
        # check if m3u8 file exists
        s3 = boto3.client('s3', config=Config(signature_version=UNSIGNED))
        prefix = "{}/hls/{}/live.m3u8".format(self.hydrophone_id, stream_id)
        objs = s3.list_objects_v2(Bucket=self.s3_bucket, Prefix=prefix, MaxKeys=1, Delimiter="/")
        
        # if it does not, exit
        if objs["KeyCount"] == 0:
            print(".m3u8 file does not exist, will retry after some time")
            return None, None, current_clip_end_time
        
        assert objs["KeyCount"] == 1

        # .m3u8 file exists so load it
        stream_obj = m3u8.load(stream_url)
        if stream_obj is None:
            return None, None, current_clip_end_time

        if stream_obj.target_duration is None:
            return None, None, current_clip_end_time

        num_total_segments = len(stream_obj.segments)
        num_segments_in_wav_duration = math.ceil(self.polling_interval/stream_obj.target_duration)

        # calculate the start index by computing the current time - start of current folder
        current_clip_end_time_unix_pst = datetime_utils.get_unix_time_from_datetime_utc(current_clip_end_time)
        time_since_folder_start = datetime_utils.get_difference_between_times_in_seconds(current_clip_end_time_unix_pst, stream_id)
        if time_since_folder_start < self.polling_interval + 20:
            # This implies that possibly a new folder was created 
            # and we do not have enough data for a 1 minute clip + 20 second buffer 
            # we exit and try again after hls polling interval
            None, None, current_clip_end_time
        
        min_num_total_segments_required = math.ceil(time_since_folder_start/stream_obj.target_duration)
        segment_start_index = min_num_total_segments_required - num_segments_in_wav_duration + 1
        segment_end_index = segment_start_index + num_segments_in_wav_duration

        if segment_end_index > num_total_segments:
            return None, None, current_clip_end_time

        # Create tmp path to hold .ts segments
        tmp_path = "tmp_path_" + self.hydrophone_id
        os.makedirs(tmp_path,exist_ok=True)

        file_names = []
        for i in range(segment_start_index, segment_end_index):
            audio_segment = stream_obj.segments[i]
            base_path = audio_segment.base_uri
            file_name = audio_segment.uri
            audio_url = base_path + file_name
            try:
                scraper.download_from_url(audio_url,tmp_path)
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

        return wav_file_path, clip_start_time, current_clip_end_time

    def is_stream_over(self):
        # returns true or false based on whether the stream is over
        # for a live stream, the stream is never over
        return False
