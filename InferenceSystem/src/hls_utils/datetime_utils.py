# utilities to perform datetime calculations
from datetime import datetime
from pytz import timezone
from datetime import timedelta

def get_clip_name_from_unix_time(source_guid, current_clip_start_time):
    """

    """

    # convert unix time to 
    readable_datetime = datetime.fromtimestamp(int(current_clip_start_time)).strftime('%Y_%m_%d_%H_%M_%S')
    clipname = source_guid + "_" + readable_datetime
    return clipname, readable_datetime

def get_difference_between_times_in_seconds(unix_time1, unix_time2):
    dt1 = datetime.fromtimestamp(int(unix_time1))
    dt2 = datetime.fromtimestamp(int(unix_time2))

    return (dt1-dt2).total_seconds()

def add_interval_to_unix_time(unix_time, interval_in_seconds):
    dt1 = datetime.fromtimestamp(int(unix_time)) + timedelta(0, interval_in_seconds)
    dt1_aware = timezone('US/Pacific').localize(dt1)
    end_time_unix = int(dt1_aware.timestamp())

    return end_time_unix

def get_unix_time_from_datetime_utc(dt_utc):

    dt_aware = timezone('UTC').localize(dt_utc)
    dt_pst = dt_aware.astimezone(timezone('US/Pacific'))

    # convert to PST
    unix_time = int(dt_pst.timestamp())

    return unix_time
