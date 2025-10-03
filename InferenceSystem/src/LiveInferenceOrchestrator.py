# Live inference orchestrator
# Rename these files
from model.podcast_inference import OrcaDetectionModel
from model.fastai_inference import FastAIModel

from orca_hls_utils.DateRangeHLSStream import DateRangeHLSStream
from orca_hls_utils.HLSStream import HLSStream

import spectrogram_visualizer
from datetime import datetime
from datetime import timedelta
from pytz import timezone

import argparse
import os
import json
import yaml

import uuid
from azure.storage.blob import BlobServiceClient, BlobClient, ContainerClient
from azure.cosmos import exceptions, CosmosClient, PartitionKey

import sys
import logging
from opencensus.ext.azure.log_exporter import AzureLogHandler
from opencensus.ext.azure.log_exporter import AzureEventHandler

AZURE_STORAGE_ACCOUNT_NAME = "livemlaudiospecstorage"
AZURE_STORAGE_AUDIO_CONTAINER_NAME = "audiowavs"
AZURE_STORAGE_SPECTROGRAM_CONTAINER_NAME = "spectrogramspng"

COSMOSDB_ACCOUNT_NAME = "aifororcasmetadatastore"
COSMOSDB_DATABASE_NAME = "predictions"
COSMOSDB_CONTAINER_NAME = "metadata"

# TODO: get this data from https://live.orcasound.net/api/json/feeds
ANDREWS_BAY_LOCATION = {"id": "rpi_andrews_bay", "name": "Andrews Bay", "longitude":  -123.1666492, "latitude": 48.5500299}
BUSH_POINT_LOCATION = {"id": "rpi_bush_point", "name": "Bush Point", "longitude":  -122.6040035, "latitude": 48.0336664}
MAST_CENTER_LOCATION = {"id": "rpi_mast_center", "name": "Mast Center", "longitude":  -122.32512, "latitude": 47.34922}
NORTH_SAN_JUAN_CHANNEL_LOCATION = {"id": "rpi_north_sjc", "name": "North San Juan Channel", "longitude":  -123.058779, "latitude": 48.591294}
ORCASOUND_LAB_LOCATION = {"id": "rpi_orcasound_lab", "name": "Orcasound Lab", "longitude":  -123.1735774, "latitude": 48.5583362}
POINT_ROBINSON_LOCATION = {"id": "rpi_point_robinson", "name": "Point Robinson", "longitude":  -122.37267, "latitude": 47.388383}
PORT_TOWNSEND_LOCATION = {"id": "rpi_port_townsend", "name": "Port Townsend", "longitude":  -122.760614, "latitude": 48.135743}
SUNSET_BAY_LOCATION = {"id": "rpi_sunset_bay", "name": "Sunset Bay", "longitude":  -122.33393605795372, "latitude": 47.86497296593844}

source_guid_to_location = {"rpi_andrews_bay": ANDREWS_BAY_LOCATION, "rpi_bush_point": BUSH_POINT_LOCATION, "rpi_mast_center": MAST_CENTER_LOCATION, "rpi_north_sjc": NORTH_SAN_JUAN_CHANNEL_LOCATION, "rpi_orcasound_lab" : ORCASOUND_LAB_LOCATION, "rpi_point_robinson": POINT_ROBINSON_LOCATION, "rpi_port_townsend" : PORT_TOWNSEND_LOCATION, "rpi_sunset_bay": SUNSET_BAY_LOCATION}

def assemble_blob_uri(container_name, item_name):

	blob_uri = "https://{acct}.blob.core.windows.net/{cont}/{item}".format(acct=AZURE_STORAGE_ACCOUNT_NAME, cont=container_name, item=item_name)
	return blob_uri


# TODO(@prgogia) read directly from example schema to prevent errors
def populate_metadata_json(
	audio_uri, image_uri,
	result_json,
	timestamp_in_iso,
	hls_polling_interval,
	model_type,
	source_guid):

	data = {}
	data["id"] = str(uuid.uuid4())
	print("===================")
	print(data["id"])

	data["modelId"]= model_type
	data["audioUri"]= audio_uri
	data["imageUri"]= image_uri
	data["reviewed"]= False
	data["timestamp"] = timestamp_in_iso
	data["whaleFoundConfidence"] = result_json["global_confidence"]
	data["location"] = source_guid_to_location[source_guid]
	data["source_guid"] = source_guid

	# whale call predictions
	local_predictions = result_json["local_predictions"]
	local_confidences = result_json["local_confidences"]

	prediction_list = []
	num_predictions = len(local_predictions)

	per_prediction_duration = hls_polling_interval/num_predictions

	# calculate segment for which
	id_num = 0
	for i in range(num_predictions):
		prediction_start_time = i*per_prediction_duration
		if local_predictions[i] == 1:
			prediction_element = {"id":id_num, "startTime": prediction_start_time, "duration": per_prediction_duration, "confidence": local_confidences[i]}
			prediction_list.append(prediction_element)
			id_num+=1

	data["predictions"] = prediction_list
	return data

def get_config_path():
	"""
	Determine the config file path based on Kubernetes namespace or command line argument.
	
	Priority:
	1. Command line --config argument (for backward compatibility and local testing)
	2. Kubernetes namespace detection with ConfigMap (for production deployments)
	
	Returns:
		str: Path to the config file
	"""
	# Check if --config argument is provided
	parser = argparse.ArgumentParser()
	parser.add_argument("--config", type=str, help="config.yml", required=False)
	parser.add_argument("--max_iterations", type=int, help="maximum number of clips to process", default=None)
	args, _ = parser.parse_known_args()
	
	if args.config:
		return args.config, args
	
	# Try to detect Kubernetes namespace
	namespace_file = "/var/run/secrets/kubernetes.io/serviceaccount/namespace"
	if os.path.exists(namespace_file):
		try:
			with open(namespace_file, "r") as f:
				namespace = f.read().strip()
			
			# Config files are mounted from ConfigMap at /config/config.yml
			config_path = f"/config/config.yml"
			
			if os.path.exists(config_path):
				print(f"Detected Kubernetes namespace: {namespace}")
				print(f"Using config from ConfigMap: {config_path}")
				return config_path, args
			else:
				raise ValueError(f"Config file not found for namespace '{namespace}' at {config_path}. Ensure ConfigMap is properly mounted.")
		except Exception as e:
			print(f"Error reading namespace or config: {e}")
			raise
	
	# If neither config argument nor namespace detection works, raise error
	raise ValueError("No config file specified. Either provide --config argument or run in Kubernetes with namespace and ConfigMap mounted at /config.")

if __name__ == "__main__":
	# Get config path
	config_path, args = get_config_path()

	# read config
	with open(config_path) as f:
		config_params = yaml.load(f, Loader=yaml.FullLoader)

	# logger to app insights
	app_insights_connection_string = os.getenv('INFERENCESYSTEM_APPINSIGHTS_CONNECTION_STRING')
	print("INSTRUMENTATION KEY: ", app_insights_connection_string)
	logger = logging.getLogger(__name__)
	if app_insights_connection_string is not None:
		logger.addHandler(AzureLogHandler(connection_string=app_insights_connection_string))
		logger.addHandler(AzureEventHandler(connection_string=app_insights_connection_string))
		logger.setLevel(logging.INFO)

	## Model Details
	model_type = config_params["model_type"]
	model_path = config_params["model_path"]
	model_local_threshold = config_params["model_local_threshold"]
	model_global_threshold = config_params["model_global_threshold"]
	# load_model_into_memory
	if model_type == "AudioSet":
		whalecall_classification_model = OrcaDetectionModel(model_path, threshold=model_local_threshold, min_num_positive_calls_threshold=model_global_threshold)
	elif model_type == "FastAI":
		model_name = config_params["model_name"]
		whalecall_classification_model = FastAIModel(model_path=model_path, model_name=model_name, threshold=model_local_threshold, min_num_positive_calls_threshold=model_global_threshold)
	else:
		raise ValueError("model_type should be one of AudioSet / FastAIModel")

	if config_params["upload_to_azure"]:
		# set up for Azure Storage Account connection
		connect_str = os.getenv('AZURE_STORAGE_CONNECTION_STRING')
		blob_service_client = BlobServiceClient.from_connection_string(connect_str)

		# set up for Azure CosmosDB connection
		cosmos_db_endpoint = "https://aifororcasmetadatastore.documents.azure.com:443/"
		cosmod_db_primary_key = os.getenv('AZURE_COSMOSDB_PRIMARY_KEY')
		client = CosmosClient(cosmos_db_endpoint, cosmod_db_primary_key)

	# create directory for wav files, metadata and spectrogram to be saved
	# get script's current dir
	local_dir = "wav_dir"
	if not os.path.exists(local_dir):
		os.makedirs(local_dir)

	## Instantiate HLSStream
	hls_stream_type = config_params["hls_stream_type"]
	hls_polling_interval = config_params["hls_polling_interval"]
	hls_hydrophone_id = config_params["hls_hydrophone_id"]
	hydrophone_stream_url = 'https://s3-us-west-2.amazonaws.com/audio-orcasound-net/' + hls_hydrophone_id
	
	if hls_stream_type == "LiveHLS":
		hls_stream = HLSStream(hydrophone_stream_url, hls_polling_interval, local_dir)
	elif hls_stream_type == "DateRangeHLS":
		hls_start_time_pst = config_params["hls_start_time_pst"]
		hls_end_time_pst = config_params["hls_end_time_pst"]

		start_dt = datetime.strptime(hls_start_time_pst, '%Y-%m-%d %H:%M')
		start_dt_aware = timezone('US/Pacific').localize(start_dt)
		hls_start_time_unix = int(start_dt_aware.timestamp())

		end_dt = datetime.strptime(hls_end_time_pst, '%Y-%m-%d %H:%M')
		end_dt_aware = timezone('US/Pacific').localize(end_dt)
		hls_end_time_unix = int(end_dt_aware.timestamp())

		hls_stream = DateRangeHLSStream(hydrophone_stream_url, hls_polling_interval, hls_start_time_unix, hls_end_time_unix, local_dir, False)
	else:
		raise ValueError("hls_stream_type should be one of LiveHLS or DateRangeHLS")

	# Adding a 10 second addition because there is logic in HLSStream that checks if now < current_clip_end_time
	current_clip_end_time = datetime.utcnow() + timedelta(0,10)

	max_iterations = args.max_iterations
	iteration_count = 0
	while not hls_stream.is_stream_over():
		if max_iterations is not None and iteration_count >= max_iterations:
			break
		iteration_count += 1

		#TODO (@prgogia) prepend hydrophone friendly name to clip and remove slashes
		clip_path, start_timestamp, current_clip_end_time = hls_stream.get_next_clip(current_clip_end_time)

		# if this clip was at the end of a bucket, clip_duration_in_seconds < 60, if so we skip it
		if clip_path:
			spectrogram_path = spectrogram_visualizer.write_spectrogram(clip_path)
			prediction_results = whalecall_classification_model.predict(clip_path)

			print("\nlocal_confidences: {}\n".format(prediction_results["local_confidences"]))
			print("local_predictions: {}\n".format(prediction_results["local_predictions"]))
			print("global_confidence: {}\n".format(prediction_results["global_confidence"]))
			print("global_prediction: {}".format(prediction_results["global_prediction"]))


			# only upload positive clip data
			if prediction_results["global_prediction"] == 1:
				print("FOUND!!!!")

				# logging to app insights
				properties = {'custom_dimensions': {'Hydrophone ID':  hls_hydrophone_id }}
				logger.info('Orca Found: ', extra=properties)

				if config_params["upload_to_azure"]:

					# upload clip to Azure Blob Storage if specified
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
					metadata = populate_metadata_json(audio_uri, spectrogram_uri, prediction_results, start_timestamp, hls_polling_interval, model_type, hls_hydrophone_id)
					database = client.get_database_client(COSMOSDB_DATABASE_NAME)
					container = database.get_container_client(COSMOSDB_CONTAINER_NAME)
					container.create_item(body=metadata)
					print("Added metadata to Azure CosmosDB")
				
			# delete local wav, spec, metadata
			if config_params["delete_local_wavs"]:
				os.remove(clip_path)
				os.remove(spectrogram_path)

			if 'log_results' in config_params:
				if config_params['upload_to_azure']:
					# override the logging, since it is already deployed
					pass
				elif config_params['log_results'] is not None:
					# log silent trial data to destination folder
					os.makedirs(config_params['log_results'], exist_ok=True)
					prediction_results.update({'model_type': config_params['model_type'],'model_name': config_params['model_name'], 'model_path':config_params['model_path']})
					json_name = os.path.basename(clip_path).replace('.wav', '.json')
					with open(os.path.join(config_params['log_results'], json_name), 'w+') as f:
						json.dump(prediction_results, f)

		# get next current_clip_end_time by adding 60 seconds to current_clip_end_time
		current_clip_end_time = current_clip_end_time + timedelta(0, hls_polling_interval)
