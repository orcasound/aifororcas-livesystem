import datetime

S3_STREAM_URLS = {
    "bush_point": "https://s3-us-west-2.amazonaws.com/audio-orcasound-net/rpi_bush_point",
    "mast_center": "https://s3-us-west-2.amazonaws.com/audio-orcasound-net/rpi_mast_center",
    "north_sjc": "https://s3-us-west-2.amazonaws.com/audio-orcasound-net/rpi_north_sjc",
    "orcasound_lab": "https://s3-us-west-2.amazonaws.com/audio-orcasound-net/rpi_orcasound_lab",
    "point_robinson": "https://s3-us-west-2.amazonaws.com/audio-orcasound-net/rpi_point_robinson",
    "port_townsend": "https://s3-us-west-2.amazonaws.com/audio-orcasound-net/rpi_port_townsend",
    "sunset_bay": "https://s3-us-west-2.amazonaws.com/audio-orcasound-net/rpi_sunset_bay",
}

# Limits time window (end - start) of negative samples to be downloaded for retraining
MAX_NEGATIVE_SAMPLE_RETRAIN_DATETIME_DELTA = datetime.timedelta(minutes=30)
