import s3_utils
import boto3
from botocore import UNSIGNED
from botocore.config import Config

import m3u8
import math
from model.scraper import download_from_url
import ffmpeg
import os
from datetime import datetime
from datetime import timedelta
from pytz import timezone
import time
import sys
from pathlib import Path

def get_clip_name_from_unix_time(source_guid, current_clip_start_time):
    """

    """

    # convert unix time to 
    readable_datetime = datetime.fromtimestamp(int(current_clip_start_time)).strftime('%Y_%m_%d_%H_%M_%S')
    clipname = source_guid + "_" + readable_datetime
    return clipname, readable_datetime

def get_difference_between_times_in_seconds(unix_time1, unix_time2):
    dt1 = datetime.fromtimestamp(int(unix_time1))
    dt2 = datetime.fromtimestamp(int(unix_time2))

    return (dt1-dt2).total_seconds()

def add_interval_to_unix_time(unix_time, interval_in_seconds):
    dt1 = datetime.fromtimestamp(int(unix_time)) + timedelta(0, interval_in_seconds)
    dt1_aware = timezone('US/Pacific').localize(dt1)
    end_time_unix = int(time.mktime(dt1_aware.timetuple()))

    return end_time_unix

#TODO: Handle date ranges that don't exist
class DateRangeHLSStream():
    """
    stream_base = 'https://s3-us-west-2.amazonaws.com/streaming-orcasound-net/rpi_orcasound_lab'
    polling_interval = 60 sec
    start_unix_time
    end_unix_time
    wav_dir
    """

    def __init__(self, stream_base, polling_interval, start_unix_time, end_unix_time, wav_dir):
        """

        """

        # Get all necessary data and create index
        self.stream_base = stream_base
        self.polling_interval_in_seconds = polling_interval
        self.start_unix_time = start_unix_time
        self.end_unix_time = end_unix_time
        self.wav_dir = wav_dir
        self.is_end_of_stream = False

        # query the stream base for all m3u8 files between the timestamps

        # split the stream base into bucket and folder
        # eg. 'https://s3-us-west-2.amazonaws.com/streaming-orcasound-net/rpi_orcasound_lab'
        # would be split into s3_bucket = 'streaming-orcasound-net' and folder_name = 'rpi_orcasound_lab'

        bucket_folder = self.stream_base.split("https://s3-us-west-2.amazonaws.com/")[1]
        tokens = bucket_folder.split("/")
        self.s3_bucket = tokens[0]
        self.folder_name = tokens[1]
        prefix = self.folder_name + "/hls/"

        # returns folder names corresponding to epochs
        all_hydrophone_folders = s3_utils.get_all_folders(self.s3_bucket, prefix=prefix)
        print("Found {} folders in all for hydrophone".format(len(all_hydrophone_folders)))

        self.valid_folders = s3_utils.get_folders_between_timestamp(all_hydrophone_folders, self.start_unix_time, self.end_unix_time)
        print("Found {} folders in date range".format(len(self.valid_folders)))

        self.current_folder_index = 0
        self.current_clip_start_time = self.start_unix_time

    def get_next_clip(self):
        """

        """

        # Get current folder
        current_folder = int(self.valid_folders[self.current_folder_index])
        clipname, clip_start_time = get_clip_name_from_unix_time(self.folder_name.replace("_", "-"), self.current_clip_start_time)

        # read in current m3u8 file
        # stream_url for the current AWS folder
        stream_url = "{}/hls/{}/live.m3u8".format(
            (self.stream_base), (current_folder))
        stream_obj = m3u8.load(stream_url)
        num_total_segments = len(stream_obj.segments)
        num_segments_in_wav_duration = math.ceil(self.polling_interval_in_seconds/stream_obj.target_duration)

        # calculate the start index by computing the current time - start of current folder
        segment_start_index = math.ceil(get_difference_between_times_in_seconds(self.current_clip_start_time, current_folder)/stream_obj.target_duration)
        segment_end_index = segment_start_index + num_segments_in_wav_duration

        if segment_end_index > num_total_segments:
            # move to the next folder and increment the current_clip_start_time to the new
            self.current_folder_index += 1
            self.current_clip_start_time = self.valid_folders[self.current_folder_index]
            return None, None

        # Can get the whole segment so update the clip_start_time for the next clip
        # We do this before we actually do the pulling in case there is a problem with this clip
        
        self.current_clip_start_time = add_interval_to_unix_time(self.current_clip_start_time, self.polling_interval_in_seconds)

        # Create tmp path to hold .ts segments
        tmp_path = "tmp_path"
        os.makedirs(tmp_path,exist_ok=True)

        file_names = []
        for i in range(segment_start_index, segment_end_index):
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
        ffmpeg.run(stream, quiet=False)

        # clear the tmp_path
        os.system(f'rm -rf {tmp_path}')

        # Get new index
        return wav_file_path, clip_start_time

    def is_stream_over(self):
        # returns true or false based on whether the stream is over
        return int(self.current_clip_start_time) >= int(self.end_unix_time)