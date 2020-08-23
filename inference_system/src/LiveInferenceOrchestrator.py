# Live inference orchestrator
from predict_and_aggregate import OrcaDetectionModel
from HLSStream import HLSStream
import spectrogram_visualizer
from datetime import datetime
from datetime import timedelta
import argparse
import os
import json

import uuid
from azure.storage.blob import BlobServiceClient, BlobClient, ContainerClient

from azure.cosmos import exceptions, CosmosClient, PartitionKey

AZURE_STORAGE_ACCOUNT_NAME = "livemlaudiospecstorage"
AZURE_STORAGE_AUDIO_CONTAINER_NAME = "audiowavs"
AZURE_STORAGE_SPECTROGRAM_CONTAINER_NAME = "spectrogramspng"

COSMOSDB_ACCOUNT_NAME = "aifororcasmetadatastore"
COSMOSDB_DATABASE_NAME = "predictions"
COSMOSDB_CONTAINER_NAME = "metadata"

# TODO(@prgogia) look up for real
ORCASOUND_LAB_LOCATION = {"id": "rpi_orcasound_lab", "name": "Haro Strait", "longitude":  -123.2166658, "latitude": 48.5499978}

def assemble_blob_uri(container_name, item_name):

    blob_uri = "https://{acct}.blob.core.windows.net/{cont}/{item}".format(acct=AZURE_STORAGE_ACCOUNT_NAME, cont=container_name, item=item_name)
    return blob_uri


# TODO(@prgogia) read directly from example schema to prevent errors
def populate_metadata_json(audio_uri, image_uri, result_json, timestamp_in_iso):
    data = {}
    data["id"] = str(uuid.uuid4())
    print("===================")
    print(data["id"])

    data["modelId"]= "PCRound2_AudioSet-fc_22"
    data["audioUri"]= audio_uri
    data["imageUri"]= image_uri
    data["reviewed"]= False
    data["timestamp"] = timestamp_in_iso
    data["whaleFoundConfidence"] = result_json["global_confidence"][0]
    data["location"] = ORCASOUND_LAB_LOCATION
    data["source_guid"] = "rpi_orcasound_lab"

    # whale call predictions
    local_predictions = result_json["local_predictions"]
    local_confidences = result_json["local_confidences"]

    prediction_list = []
    num_predictions = len(local_predictions)
    id_num = 0
    for i in range(num_predictions):
        prediction_start_time = i*2.45
        if local_predictions[i] == 1:
            prediction_element = {"id":id_num, "startTime": prediction_start_time, "duration": 2.45, "confidence": local_confidences[i]}
            prediction_list.append(prediction_element)
            id_num+=1

    data["predictions"] = prediction_list
    return data

if __name__ == "__main__":

    # load_model_into_memory
    model_path = "model"
    whalecall_classification_model = OrcaDetectionModel(model_path, 0.9)

    whale_found_confidence_write_threshold = 0.5

    # create directory for wav files, metadata and spectrogram to be saved
    # get script's current dir
    local_dir = "wav_dir"
    if not os.path.exists(local_dir):
        os.makedirs(local_dir)

    # set up for Azure Storage Account connection
    connect_str = os.getenv('AZURE_STORAGE_CONNECTION_STRING')
    blob_service_client = BlobServiceClient.from_connection_string(connect_str)

    # set up for Azure CosmosDB connection
    cosmos_db_endpoint = "https://aifororcasmetadatastore.documents.azure.com:443/"
    cosmod_db_primary_key = os.getenv('AZURE_COSMOSDB_PRIMARY_KEY')
    client = CosmosClient(cosmos_db_endpoint, cosmod_db_primary_key)

    # instantiate HLSStream
    # TODO (@prgogia) insert mock data into HLS stream for demo purposes
    Orcasound_lab_url = 'https://s3-us-west-2.amazonaws.com/streaming-orcasound-net/rpi_orcasound_lab'
    hls_stream = HLSStream(Orcasound_lab_url, 60, local_dir)

    # Adding a 10 second addition because there is logic in HLSStream that checks if now < current_clip_end_time
    current_clip_end_time = datetime.utcnow() + timedelta(0,10)

    while True:
        #TODO (@prgogia) prepend hydrophone friendly name to clip
        clip_path, start_timestamp = hls_stream.get_next_clip(current_clip_end_time)

        # if this clip was at the end of a bucket, clip_duration_in_seconds < 60, if so we skip it
        if clip_path:
            spectrogram_path = spectrogram_visualizer.write_spectrogram(clip_path)
            prediction_results = whalecall_classification_model.predict(clip_path)

            #TODO(@prgogia) 
            # only keep track of positive data
            # if prediction_results["global_prediction"] == 1:

            # upload clip to Azure Blob Storage
            audio_clip_name = os.path.basename(clip_path)
            audio_blob_client = blob_service_client.get_blob_client(container=AZURE_STORAGE_AUDIO_CONTAINER_NAME, blob=audio_clip_name)
            with open(clip_path, "rb") as data:
                audio_blob_client.upload_blob(data)
            audio_uri = assemble_blob_uri(AZURE_STORAGE_AUDIO_CONTAINER_NAME, audio_clip_name)
            print("Uploaded audio to Azure Storage")

            # upload spectrogram to Azure Blob Storage
            spectrogram_name = os.path.basename(spectrogram_path)
            spectrogram_blob_client = blob_service_client.get_blob_client(container=AZURE_STORAGE_SPECTROGRAM_CONTAINER_NAME, blob=spectrogram_name)
            with open(spectrogram_path, "rb") as data:
                spectrogram_blob_client.upload_blob(data)
            spectrogram_uri = assemble_blob_uri(AZURE_STORAGE_SPECTROGRAM_CONTAINER_NAME, spectrogram_name)
            print("Uploaded spectrogram to Azure Storage")

            # Insert metadata into CosmosDB
            metadata = populate_metadata_json(audio_uri, spectrogram_uri, prediction_results, start_timestamp)
            database = client.get_database_client(COSMOSDB_DATABASE_NAME)
            container = database.get_container_client(COSMOSDB_CONTAINER_NAME)
            container.create_item(body=metadata)
            print("Added metadata to Azure CosmosDB")
                
            # delete local wav, spec, metadata
            os.remove(clip_path)
            os.remove(spectrogram_path)

        # get next current_clip_end_time by adding 60 seconds to current_clip_end_time
        current_clip_end_time = current_clip_end_time + timedelta(0, 60)
        
