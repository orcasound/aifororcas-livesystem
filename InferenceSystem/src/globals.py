import datetime

S3_STREAM_URLS = {
    'orcasound_lab':'https://s3-us-west-2.amazonaws.com/streaming-orcasound-net/rpi_orcasound_lab',
    'port_townsend':'https://s3-us-west-2.amazonaws.com/streaming-orcasound-net/rpi_port_townsend',
    'bush_point':'https://s3-us-west-2.amazonaws.com/streaming-orcasound-net/rpi_bush_point'
}

MAX_DATETIME_DELTA = datetime.datetime.timedelta(minutes=30)