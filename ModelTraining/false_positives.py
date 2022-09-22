from azure.cosmos import CosmosClient
from azure.identity import DefaultAzureCredential
import json

COSMOS_DB_NAME = "predictions"
COSMOS_CONTAINER_NAME = "metadata"


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
        print(json.dumps(metadata_item, indent=True))

def _download_false_positive_samples(metadata_item, out_folder):
    pass


if __name__ == "__main__":
    get_false_positives('2022-05-05', '2022-09-20', 0)
