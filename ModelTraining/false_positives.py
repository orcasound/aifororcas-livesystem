from azure.cosmos import CosmosClient
from azure.identity import DefaultAzureCredential
import json
import requests
from pathlib import Path
import pydub
import tempfile

COSMOS_DB_NAME = "predictions"
COSMOS_CONTAINER_NAME = "metadata"

# Main method
# Downloads audio clips (*.wav) of false positives from OrcaHellow CosmosDB
# Each OrcaHello clip is ~60seconds, but we'll want to separate out each individual, marked false positive
def get_false_positives(start_date, end_date, out_folder):
    app_settings = None
    with open("appsettings.development.json") as appsettings_file:
        appsettings = json.load(appsettings_file)

    cosmos_client = CosmosClient(
        appsettings["cosmos_account_uri"], appsettings["cosmos_account_pkey"]
    )
    database = cosmos_client.get_database_client(COSMOS_DB_NAME)
    container = database.get_container_client(COSMOS_CONTAINER_NAME)

    query = f"""
    SELECT items.id, items.location.name, items.location.latitude, items.location.longitude, items.timestamp,
        items.audioUri, items.predictions
    FROM items
    WHERE items.reviewed = true AND
    items.SRKWFound = 'no' AND
    items.timestamp >= '{start_date}' AND
    items.timestamp < '{end_date}'
    ORDER by items.timestamp
    """

    for metadata_item in container.query_items(
        query=query, enable_cross_partition_query=True
    ):
        _download_false_positive_samples(metadata_item, out_folder)


# TODO Complete this method
#       Splice the orcahello_sample into the various predictions in the metadata_item
#           Each prediction has start time and duration
#           Pydub library allows us to splice (in millisecond intervals) the .wav file
def _download_false_positive_samples(metadata_item, out_folder):
    Path(out_folder).mkdir(exist_ok=True)
    tf = tempfile.NamedTemporaryFile()
    dl_uri = metadata_item["audioUri"]
    tf.write(requests.get(dl_uri).content)
    orcahello_sample = pydub.AudioSegment.from_wav(tf.name)

    for prediction in metadata_item["predictions"]:
        dl_filename = Path(dl_uri).name
        out_filepath = Path(out_folder, f"{dl_filename}-{prediction['id']}")
        print(f"Saving {dl_filename} to {out_filepath}")


if __name__ == "__main__":
    get_false_positives("2022-09-15", "2022-09-18", "out")
