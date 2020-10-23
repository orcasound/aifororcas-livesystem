#  script to convert files into a dataset

import os
import argparse
import glob
import json

inputDir = r"/Users/prakrutigogia/Documents/Microsoft/AlwaysBeLearning/MSHack/Round7"
dataset = "podcast_round7"
location = "orcasound_lab"

all_jsons = glob.glob(inputDir + "/*.json")
all_wavs = glob.glob(inputDir + "/round7/*.wav")
tsv_file = os.path.join(inputDir, "annotations.tsv")
if os.path.exists(tsv_file):
    os.remove(tsv_file)

with open(tsv_file, "a+") as t:
    header = "dataset\twav_filename\tstart_time_s\tduration_s\tlocation\tdate\tpst_or_master_tape_identifier\n"
    t.write(header)

    for wav in all_wavs:
       wav_filename =  os.path.basename(wav)
       date_tokens  = os.path.splitext(wav_filename)[0].split("_")
       date = date_tokens[1] + "-" + date_tokens[2] + "-" + date_tokens[3]
       pst = date_tokens[4] + ":" + date_tokens[5] + ":" + date_tokens[6]
       line = dataset + "\t" + wav_filename + "\t" + "0.0" + "\t" + "0.0" + "\t" + location + "\t" + date + "\t" + pst + "\n"
       t.write(line)

    # for annotation_json in all_jsons:
    #     with open(annotation_json, "r") as f:
    #         data = json.load(f)
    #         wav_filename = os.path.basename(data["uri"])
    #         date_tokens = data["absolute_time"].split("_")
    #         pst = date_tokens[3] + ":" + date_tokens[4] + ":" + date_tokens[5]
    #         if len(data["annotations"]) == 0:
    #             line = dataset + "\t" + wav_filename + "\t" + "0.0" + "\t" + "0.0" + "\t" + location + "\t" + date + "\t" + pst + "\n"
    #             t.write(line)
    #         else:
    #             for annotation in data["annotations"]:
    #                 line = dataset + "\t" + wav_filename + "\t" + str(annotation["start_s"]) + "\t" + str(annotation["duration_s"]) + "\t" + location + "\t" + date + "\t" + pst + "\n"
    #                 t.write(line)
