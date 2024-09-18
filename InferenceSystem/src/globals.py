import datetime

S3_STREAM_URLS = {
<<<<<<< HEAD
    "orcasound_lab": "https://s3-us-west-2.amazonaws.com/audio-orcasound-net/rpi_orcasound_lab",
    "port_townsend": "https://s3-us-west-2.amazonaws.com/audio-orcasound-net/rpi_port_townsend",
    "bush_point": "https://s3-us-west-2.amazonaws.com/audio-orcasound-net/rpi_bush_point",
    "sunset_bay": "https://s3-us-west-2.amazonaws.com/audio-orcasound-net/rpi_sunset_bay",
    "point_robinson": "https://s3-us-west-2.amazonaws.com/audio-orcasound-net/rpi_point_robinson",
=======
    "orcasound_lab": "https://s3-us-west-2.amazonaws.com/streaming-orcasound-net/rpi_orcasound_lab",
    "port_townsend": "https://s3-us-west-2.amazonaws.com/streaming-orcasound-net/rpi_port_townsend",
    "bush_point": "https://s3-us-west-2.amazonaws.com/streaming-orcasound-net/rpi_bush_point",
    "sunset_bay": "https://s3-us-west-2.amazonaws.com/streaming-orcasound-net/rpi_sunset_bay",
    "point_robinson": "https://s3-us-west-2.amazonaws.com/streaming-orcasound-net/rpi_point_robinson",
    "mast_center": "https://s3-us-west-2.amazonaws.com/streaming-orcasound-net/rpi_mast_center",
    "north_sjc": "https://s3-us-west-2.amazonaws.com/streaming-orcasound-net/rpi_north_sjc"
>>>>>>> ed567fc (Add in references for Mast Center and North San Juan Center)
}

# Limits time window (end - start) of negative samples to be downloaded for retraining
MAX_NEGATIVE_SAMPLE_RETRAIN_DATETIME_DELTA = datetime.timedelta(minutes=30)
