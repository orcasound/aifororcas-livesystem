
import argparse
import datetime
import globals

def main():
    """
    Example usage: 
    python3 PrepareDataForPredictionExplorer.py --start_time "2020-07-25 19:15" --end_time "2020-07-25 20:15" --dataset_folder /Users/prakrutigogia/Documents/Microsoft/AlwaysBeLearning/MSHack/Round4
    """

    parser = argparse.ArgumentParser()
    parser.add_argument('--start_time', type=str, required=True, \
        help='Start time in PST in following format 2020-07-25 19:15')
    parser.add_argument('--end_time', type=str, required=True, \
        help='End time in PST in following format 2020-07-25 20:15')
    parser.add_argument('hydrophone', choices=list(globals.S3_STREAM_URLS.keys()), required=True, \
        help='Hydrophone location')

    args = parser.parse_args()

    datetime_format = '%Y-%m-%d %H:%M:%S'
    start_datetime = datetime.datetime.strptime(args.start_time, datetime_format)
    end_datetime = datetime.datetime.strptime(args.end_time, datetime_format)

    assert(end_datetime - start_datetime <= globals.MAX_DATETIME_DELTA)
    assert(args.s3_stream in globals.S3_STREAM_URLS)

    #CALL PreProcessor function
        # Param :: Stream

    #CALL blending function
        # Param  :: Model pathname (provide default if not in config file)

    # CALL into Aayush's forwardProcessing script [forward_finetune]
        # Param  :: Path to latest training data (provide default NONE until in config file)
        # Return :: (Model which we can upload as latest model to ..., new path to new training data set)

    # Upload new model version
 
    # Upload new training set

    # Test new trained model with current evaluation benchmark and Output graphs and save all to auto-PR 

if __name__ == "__main__":
    main()