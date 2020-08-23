#! /usr/bin/env python3

from azure.cosmosdb.table.tableservice import TableService
from datetime import datetime
import json

DB_ACCOUNT = "20200125hackday"
DB_KEY = "PLACEHOLDER"
DB_TABLE_NAME = "predictions"

def dump2db(
        json_obj,
        partition_key = datetime.today().strftime("%Y-%m-%d"),
        row_key = datetime.today().strftime("%H-%M-%S")):
    table_service = TableService(
        account_name=DB_ACCOUNT,
        account_key=DB_KEY)

    json_obj['PartitionKey'] = partition_key
    json_obj['RowKey'] = row_key

    # Azure table storage doesn't handle list or object
    # we'll need to migrate to Cosmos DB if we need those
    for key in json_obj:
        if isinstance(json_obj[key], list) or isinstance(json_obj[key], dict):
            json_obj[key] = json.dumps(json_obj[key])

    print(json_obj)
    table_service.insert_or_replace_entity(
        DB_TABLE_NAME,
        json_obj)
