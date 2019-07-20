import m3u8
import numpy as np
import os
import random
import time
import uuid
import urllib.request
import pdb
from pathlib import Path # TODO: make path usage a bit more consistent
from scraper import download_from_url

make_osl_url = lambda x: "https://s3-us-west-2.amazonaws.com/streaming-orcasound-net/rpi_orcasound_lab/hls/{}/live.m3u8".format(x)

make_bush_url = lambda x: "https://s3-us-west-2.amazonaws.com/streaming-orcasound-net/rpi_bush_point/hls/{}/live.m3u8".format(x)

urls = []
urls.extend([ make_osl_url(u) for u in ["1541140334","1541161938","1543256233","1543613535"] ])
urls.extend([ make_bush_url(u) for u in ["1539455461","1539498651","1540881062","1545741018"] ])

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
