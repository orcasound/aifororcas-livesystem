import argparse
import datetime
import globals


def main():
    """
    Example usage:
    python3 model_retrain_negative_samples.py --start_time "2020-07-25 19:15:00" --end_time "2020-07-25 19:45:00" --hydrophone "bush_point"
    """

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--start_time",
        type=str,
        required=True,
        help="Start time in PST in following format 2020-07-25 19:15:00",
    )
    parser.add_argument(
        "--end_time",
        type=str,
        required=True,
        help="End time in PST in following format 2020-07-25 20:15:00. Cannot be greater than (start_time + 30 minutes)",
    )
    parser.add_argument(
        "--hydrophone",
        choices=list(globals.S3_STREAM_URLS.keys()),
        required=True,
        help="Hydrophone location",
    )

    args = parser.parse_args()

    datetime_format = "%Y-%m-%d %H:%M:%S"
    start_datetime = datetime.datetime.strptime(args.start_time, datetime_format)
    end_datetime = datetime.datetime.strptime(args.end_time, datetime_format)

    assert (
        end_datetime - start_datetime
    ) <= globals.MAX_NEGATIVE_SAMPLE_RETRAIN_DATETIME_DELTA

    # CALL PreProcessor function
    # Param :: Stream

    # CALL blending function
    # Param  :: Model pathname (provide default if not in config file)

    # CALL into Aayush's forwardProcessing script [forward_finetune]
    # Param  :: Path to latest training data (provide default NONE until in config file)
    # Return :: (Model which we can upload as latest model to ..., new path to new training data set)

    # Upload new model version

    # Upload new training set

    # Test new trained model with current evaluation benchmark and Output graphs and save all to auto-PR


if __name__ == "__main__":
    main()
