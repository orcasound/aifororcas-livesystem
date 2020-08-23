# MockHLSStream class 
import urllib.request
import os
import m3u8
import math
import ffmpeg #ffmpeg-python
from model.scraper import download_from_url
from datetime import datetime
import time
from pathlib import Path
import glob
import shutil
import random

#TODO (@prgogia) Handle errors due to rebooting of hydrophone
class MockHLSStream():
    """
    Takes in as input

    true_srkw_folder = ""
    polling_interval = 60 sec
    true_srkw_probability_threshold = 0.9
    For randomly chosen minutes, it sends audio elements from the folder
    """

    def __init__(self, true_srkw_folder, polling_interval, true_srkw_probability_threshold, wav_dir):
        self.true_srkw_folder = true_srkw_folder
        self.polling_interval = polling_interval
        self.true_srkw_probability_threshold = true_srkw_probability_threshold
        self.wav_dir = wav_dir

        search_string = os.path.join(self.true_srkw_folder, "*.wav")
        self.true_srkw_files = glob.glob(search_string)
        self.num_true_srkw_files = len(self.true_srkw_files)

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
        print("Listening to location {loc}".format(loc=self.true_srkw_folder))
        
        # generate a random number between 0 and 1
        random_probability = random.uniform(0, 1)
        if random_probability > self.true_srkw_probability_threshold:
            # This is an SRKW clip
            # randomly pick one of the files
            srkw_file_num = random.randint(0, self.num_true_srkw_files-1)
            selected_srkw_file = self.true_srkw_files[srkw_file_num]
            print(selected_srkw_file)
            
            # copy file to wav_dir and rename
            wav_file_path = os.path.join(self.wav_dir, clipname + ".wav")
            shutil.copy(selected_srkw_file, wav_file_path)
        else:
            None, None

        return wav_file_path, clipname

