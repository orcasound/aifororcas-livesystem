#! /usr/bin/env python3
import boto3
from botocore import UNSIGNED
from botocore.config import Config

# Borrowed pagination code from https://alexwlchan.net/2019/07/listing-s3-keys/
def get_all_folders(bucket: str, prefix: str) -> [str]:
    """
    Generate objects in an S3 bucket.

    :param bucket: Name of the S3 bucket.
    :param prefix: Only fetch objects whose key starts with
        this prefix (optional).
    :param suffix: Only fetch objects whose keys end with
        this suffix (optional).
    """
    s3 = boto3.client('s3', config=Config(signature_version=UNSIGNED))
    paginator = s3.get_paginator("list_objects_v2")

    kwargs = {'Bucket': bucket, 'Prefix': prefix, 'Delimiter': "/"}

    all_keys = []
    # Orcasound buckets are not predictably spaced throughout the time that they've been up some are 2 hours, some hold 24 hours
    # So not making any assumptions

    for page in paginator.paginate(**kwargs):
        try:
            common_prefixes = page["CommonPrefixes"]
            prefixes = [prefix["Prefix"].split("/")[-2] for prefix in common_prefixes]
            all_keys.extend(prefixes)

        except KeyError:
            print("No content returned")
            break

    return all_keys

def get_folders_between_timestamp(bucket_list: [str], start_time: str, end_time: str) -> [int]:
    bucket_list = [int(bucket) for bucket in bucket_list]
    start_index = 0
    end_index = len(bucket_list) - 1
    while int(bucket_list[start_index]) < int(start_time):
        start_index += 1
    while int(bucket_list[end_index]) > int(end_time):
        end_index -= 1
    return bucket_list[start_index-1:end_index + 1]