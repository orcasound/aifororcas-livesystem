# script to create data for prediction explorer

# input: start_time, end_time
import argparse
import sys
from datetime import datetime
from pytz import timezone
from model.podcast_inference import OrcaDetectionModel
from model.fastai_inference import FastAIModel
import globals

import json
import os
from orca_hls_utils.DateRangeHLSStream import DateRangeHLSStream
from pathlib import Path

from azure.storage.blob import BlobServiceClient, BlobClient, ContainerClient
from azure.cosmos import exceptions, CosmosClient, PartitionKey

AUDIO_BLOB_STORAGE_ACCOUNT = "mldevdatastorage"
PODCAST_AZURE_AUDIO_CONTAINER = "podcast-audiosegments"
PODCAST_AZURE_PREDICTIONS_CONTAINER_PREFIX = "podcast-predictions-"
PODCAST_AZURE_ANNOTATIONS_CONTAINER_PREFIX = "podcast-annotations-"
HOP_S = 1.00
DURATION_S = 2.00

def assemble_blob_uri(container_name, round_id, item_name):

    blob_uri = "https://{acct}.blob.core.windows.net/{cont}/{round_id}/{item}".format(acct=AUDIO_BLOB_STORAGE_ACCOUNT, cont=container_name, round_id = round_id, item=item_name)
    return blob_uri

def write_annotations_to_tsv(tsv_full_filename, clip_name, local_confidences, annotation_threshold):
    """
    """
    with open(tsv_full_filename, "a") as f:
        num_predictions = len(local_confidences)

        for i in range(num_predictions):
            prediction_start_time = i*HOP_S
            confidence = local_confidences[i]
            if confidence > annotation_threshold:
                f.write("{wav}\t{start}\t{duration}\t{confidence}\n".format(\
                    wav=clip_name, start=prediction_start_time, duration=DURATION_S, confidence=confidence))
            
def upload_audio_and_predictions(clip_path, clip_start_time, source_guid, local_confidences, annotation_threshold, round_id, blob_service_client, dataset_folder):
    """

    """
    
    audio_uri = assemble_blob_uri(PODCAST_AZURE_AUDIO_CONTAINER, round_id, os.path.basename(clip_path))

    data = {}
    data["uri"] = audio_uri
    data["absolute_time"] = clip_start_time
    data["source_guid"] = source_guid

    prediction_list = []
    num_predictions = len(local_confidences)
    for i in range(num_predictions):
        prediction_start_time = i*HOP_S
        if local_confidences[i] > annotation_threshold:
            prediction_element = {"start_time_s": prediction_start_time, "duration_s": DURATION_S, "confidence": local_confidences[i]}
            prediction_list.append(prediction_element)

    data["annotations"] = prediction_list
    data["model_guid"] = "PodCast_Round3"

    # we don't want to save clips for annotation where we have no valid predictions
    clip_basename = os.path.basename(clip_path)
    predictions_filename = os.path.splitext(clip_basename)[0]
    if len(prediction_list) > 0:
        predictions_json_full_path = os.path.join(dataset_folder, predictions_filename + ".json")
        with open(predictions_json_full_path, "w") as f:
            json.dump(data, f)

        # upload audio to blob storage
        audio_clip_name = os.path.basename(clip_path)
        audio_clip_with_round = os.path.join(round_id, audio_clip_name)
        audio_clip_container_client = blob_service_client.get_container_client(container=PODCAST_AZURE_AUDIO_CONTAINER)

        try:
            audio_clip_container_client.get_container_properties()
        except Exception as e:
            print(e)
            print("Container did not exist, creating now")
            audio_clip_container_client.create_container()

        audio_blob_client = blob_service_client.get_blob_client(container=PODCAST_AZURE_AUDIO_CONTAINER, blob=audio_clip_with_round)
        try:
            with open(clip_path, "rb") as data:
                audio_blob_client.upload_blob(data)
        except Exception as e:
            print(e)
            print("Audio blob already existed")

        # TODO (@prgogia)
        # upload annotations to blob storage
        predictions_name = os.path.basename(predictions_json_full_path)

        predictions_container_name = PODCAST_AZURE_PREDICTIONS_CONTAINER_PREFIX + round_id
        predictions_container_client = blob_service_client.get_container_client(container=predictions_container_name)

        try:
            predictions_container_client.get_container_properties()
        except Exception as e:
            print(e)
            print("Container did not exist, creating now")
            predictions_container_client.create_container()

        predictions_blob_client = blob_service_client.get_blob_client(container=predictions_container_name, blob=predictions_name)

        try:
            with open(predictions_json_full_path, "rb") as data:
                predictions_blob_client.upload_blob(data)
        except Exception as e:
            print(e)
            print("Predictions blob existed")

def create_dataset_from_unix_daterange(start_time_unix, end_time_unix, s3_stream, model_path, annotation_threshold, round_id, dataset_folder):
    """

    """

    start_unix_str = str(start_time_unix)
    end_unix_str = str(end_time_unix)
    # whalecall_classification_model = OrcaDetectionModel(model_path, 0.0)
    model_name = Path(model_path).name 
    model_path = Path(model_path).parent  
    whalecall_classification_model = FastAIModel(
        model_path=model_path, model_name=model_name, 
        threshold=annotation_threshold, min_num_positive_calls_threshold=1
    )
    hlsstream = DateRangeHLSStream(s3_stream, 60, start_unix_str, end_unix_str, dataset_folder)

    tsv_full_filename = os.path.join(dataset_folder, "predictions.tsv")
    # add a header for the tsv file
    with open(tsv_full_filename, "w+") as f:
        f.write("wav_filename\tstart_time_s\tduration_s\tconfidence\n")

    # blob_service_client
    connect_str = os.getenv('PODCAST_AZURE_STORAGE_CONNECTION_STRING')
    print(connect_str)
    blob_service_client = BlobServiceClient.from_connection_string(connect_str)

    # create container to hold annotations
    annotations_container_name = PODCAST_AZURE_ANNOTATIONS_CONTAINER_PREFIX + round_id
    annotations_container_client = blob_service_client.get_container_client(container=annotations_container_name)

    try:
        annotations_container_client.get_container_properties()
    except Exception as e:
        print(e)
        print("Container did not exist, creating now")
        annotations_container_client.create_container()

    while not hlsstream.is_stream_over():

        print("Trying to get next clip")
        clip_path, _, clip_name = hlsstream.get_next_clip()

        # if this clip was at the end of a bucket, clip_duration_in_seconds < 60, if so we skip it
        if clip_path:
            prediction_results = whalecall_classification_model.predict(clip_path)

            # Write tsv and wav dir for prediction explorer
            # TSV format is the following
            # wav_filename	start_time_s	duration_s
            local_confidences = prediction_results["local_confidences"]

            write_annotations_to_tsv(tsv_full_filename, os.path.basename(clip_path) , local_confidences, annotation_threshold)
            upload_audio_and_predictions(clip_path, clip_name, s3_stream, local_confidences, annotation_threshold, round_id, blob_service_client, dataset_folder)


    # TODO Create data statistics, num minutes with positives and negatives

def create_prediction_explorer_dataset(start_time_str, end_time_str, s3_stream, model_path, annotation_threshold, round_id, dataset_folder):
    """

    """

    # 2020-07-25 19:50
    start_dt = datetime.strptime(start_time_str, '%Y-%m-%d %H:%M')
    start_dt_aware = timezone('US/Pacific').localize(start_dt)
    start_time_unix = int(start_dt_aware.timestamp())

    # 2020-07-25 20:15
    end_dt = datetime.strptime(end_time_str, '%Y-%m-%d %H:%M')
    end_dt_aware = timezone('US/Pacific').localize(end_dt)
    end_time_unix = int(end_dt_aware.timestamp())
    
    print("Downloading HLS data from start unix time : {start} to end unit time : {end}".format(start=start_time_unix, end=end_time_unix))

    create_dataset_from_unix_daterange(start_time_unix, end_time_unix, s3_stream, model_path, annotation_threshold, round_id, dataset_folder)

def main():
    """
    Example usage: 
    python3 PrepareDataForPredictionExplorer.py --start_time "2020-07-25 19:15" --end_time "2020-07-25 20:15" --dataset_folder /Users/prakrutigogia/Documents/Microsoft/AlwaysBeLearning/MSHack/Round4
    """

    parser = argparse.ArgumentParser()
    parser.add_argument("--start_time", type=str, required=True, \
        help="Start time in PST in following format 2020-07-25 19:15")
    parser.add_argument("--end_time", type=str, required=True, \
        help="End time in PST in following format 2020-07-25 20:15")
    # TODO: get list of streams from https://live.orcasound.net/api/json/feeds instead of hard coding it
    parser.add_argument("--s3_stream", type=str, required=True, \
        help="Hydrophone stream (bush_point/mast_center/north_sjc/orcasound_lab/point_robinson/port_townsend/sunset_bay)")
    parser.add_argument("--model_path", type=str, required=True, \
        help="Path to the model folder that contains weights and mean, invstd")
    parser.add_argument("--annotation_threshold", type=float, required=True, \
        help="The threshold Eg. 0.4")
    parser.add_argument("--round_id", type=str, required=True, \
        help="Classify this time period as a round Eg. round4")
    parser.add_argument("--dataset_folder", type=str, required=True, \
        help="Full path to local directory to save dataset")

    args = parser.parse_args()

    assert args.s3_stream in globals.S3_STREAM_URLS

    # Writes the dataset to args.dataset_folder
    dataset_folder = Path(args.dataset_folder) / args.round_id
    if not os.path.exists(dataset_folder):
        os.makedirs(dataset_folder)
    
    create_prediction_explorer_dataset(
        args.start_time, args.end_time, globals.S3_STREAM_URLS[args.s3_stream],
        args.model_path, args.annotation_threshold,
        args.round_id, dataset_folder
    )

if __name__ == "__main__":
    main()
