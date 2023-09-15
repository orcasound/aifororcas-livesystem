# Get Started
# pip install ffmpeg-python python-dateutil
# AND
# winget install Gyan.FFmpeg
# you may need to add the FFmpeg path to system/user PATH variable
# "C:\Users\User\AppData\Local\Microsoft\WinGet\Links\ffmpeg.exe"
# or
# "C:\Users\User\AppData\Local\Microsoft\WinGet\Packages\Gyan.FFmpeg_Microsoft.Winget.Source_8wekyb3d8bbwe\ffmpeg-6.0-full_build\bin\ffmpeg.exe"

# USAGE
# python .\awsAudioTS2Mp3DownloadandConvert.py --date '2020-09-11 22:14:00 PST' --node rpi_orcasound_lab
# AWS Bucket defaults to: streaming-orcasound-net. Otherwise use --awsBucket streaming-orcasound-net to change bucket name

# To-Do
# Select a subset of the ts files to download based on the delta from the epoch datetime and user input datetime. This reduces the download and output file size.
# Rename the output file

# ffmpeg -i '.\live.m3u8' -c copy -bsf:a aac_adtstoasc demo.mp4

import datetime, argparse, os, requests, ffmpeg, shutil
from dateutil import parser
from glob import glob

import boto3
from botocore import UNSIGNED
from botocore.client import Config

locations = [
    'rpi_bush_point',
    'rpi_mast_center',
    'rpi_north_sjc',
    'rpi_orcasound_lab',
    'rpi_port_townsend',
    'rpi_sunset_bay'
]

CLI=argparse.ArgumentParser()
CLI.add_argument(
    "--date",
    nargs=1,
    type=str,
    default=None,
    help="The data of the audio data of interest"
)
CLI.add_argument(
    "--node",
    nargs=1,
    type=str,
    default='rpi_orcasound_lab',
    help="The name of the node where the hydrophone is located. Default: rpi_orcasound_lab"
)
CLI.add_argument(
    "--awsBucket",
    nargs=1,
    type=str,
    default='streaming-orcasound-net',
    help="The name of the AWS audio data is stored. Default: streaming-orcasound-net"
)
args = CLI.parse_args()

assert(args.node in locations)

s3_client = boto3.client('s3', config=Config(signature_version=UNSIGNED, region_name='us-east-1'))

DELETE_TEMP_FILES = True
user_directory = os.environ['USERPROFILE']
temp_dir_name = os.path.join('AppData\Local\Temp', 'orca_ffmpeg_temp')
temp_directory_path = os.path.join(user_directory, temp_dir_name)
os.makedirs(temp_directory_path, exist_ok=True)
streamingBucketURL = "https://streaming-orcasound-net.s3.amazonaws.com/"

if args.date and len(args.date) == 1:
    user_input = args.date[0]
elif len(args.date) > 1:
    assert(len(args.date) == 1) # just placeholder failure
else:
    user_input = input('Enter Date in format: YYYY-MM-DD HH:MM:SS TZ. Ex: 2020-09-11 22:14:00 PST')
    # user_input = '2020-09-11 22:14:00 PST' #'2023-09-12 02:12:00'
if args.node:
    node_name = args.node
else:
    node_name = 'rpi_orcasound_lab'
if args.awsBucket:
    my_bucket = args.awsBucket
else:
    my_bucket = 'streaming-orcasound-net'
user_input_formatted = parser.parse(user_input)
# user_input_formatted = time.strptime(user_input, '%Y-%m-%d %H:%M:%S %Z')
print('time_formatted', user_input_formatted)
user_input_epoch = user_input_formatted.timestamp()
print('user_input_epoch', user_input_epoch)


def get_all_aws_objects(bucket_name, filename_prefix):
    '''Get all files containing the prefix provided'''
    all_object_keys = [e['Key'] for p in s3_client.get_paginator("list_objects_v2")\
        .paginate(Bucket=bucket_name, Prefix=filename_prefix)
        for e in p['Contents']]
    
    return all_object_keys


def download(url: str, dest_folder: str):
    # https://stackoverflow.com/questions/56950987/download-file-from-url-and-save-it-in-a-folder-python
    if not os.path.exists(dest_folder):
        os.makedirs(dest_folder)  # create folder if it does not exist

    filename = url.split('/')[-1].replace(" ", "_")  # be careful with file names
    file_path = os.path.join(dest_folder, filename)

    r = requests.get(url, stream=True)
    if r.ok:
        print("saving to", os.path.abspath(file_path))
        with open(file_path, 'wb') as f:
            for chunk in r.iter_content(chunk_size=1024 * 8):
                if chunk:
                    f.write(chunk)
                    f.flush()
                    os.fsync(f.fileno())
    else:  # HTTP status code 4XX/5XX
        print("Download failed: status code {}\n{}".format(r.status_code, r.text))

filename_prefix = node_name + '/hls/' + str(user_input_epoch)[:4]

# Get all files containing the prefix provided
aws_files = get_all_aws_objects(bucket_name=my_bucket, filename_prefix=filename_prefix)

# Filter out files to only retain directories (epoch folders)
aws_epochs = [aws_file.split('hls/')[-1].split('/')[0] for aws_file in aws_files]
aws_epochs = sorted(list(set(aws_epochs)))
print('aws_epochs', aws_epochs)
current_epoch = ''
epoch_needed = ''
# find the epoch folder that contains the epoch (i.e. timestamp) being requested
for aws_epoch in aws_epochs:
    if current_epoch:
        last_epoch = current_epoch
    current_epoch = aws_epoch
    if int(current_epoch) > int(user_input_epoch):
        print(current_epoch, '>', str(int(user_input_epoch)))
        epoch_needed = last_epoch
        break
    
if epoch_needed:
    print('Epoch needed', epoch_needed)
else:
    print("No Matching Epoch Found")
assert(epoch_needed)
epoch_folder = datetime.datetime.fromtimestamp(int(epoch_needed))
print('Conversion to datetime', epoch_folder)
print(int(user_input_epoch) - int(epoch_needed))

# get files from desired epoch folder
filename_prefix = node_name + '/hls/' + epoch_needed
aws_files = get_all_aws_objects(bucket_name=my_bucket, filename_prefix=filename_prefix)

# download the files locally to a temp folder
for aws_file in aws_files:
    if '.' in aws_file:
        download(streamingBucketURL+aws_file, dest_folder=temp_directory_path)

glob_path = os.path.join(temp_directory_path, '*.ts')
glob_list_files = glob(glob_path)
for glob_list_file in glob_list_files:
    filename = os.path.basename(glob_list_file)
    if len(filename) <= 10:
        new_filename = filename.replace('live', 'live0')
        new_path = os.path.join(temp_directory_path, new_filename)
        shutil.move(glob_list_file, new_path)
glob_list_files = glob(glob_path)
# for file in live*; do mv "$file" "${file#live}"; done;
# for i in *.ts ; do
#     mv $i `printf '%04d' ${i%.ts}`.ts
# done
# printf "file '%s'\n" ./*.ts > mylist.txt
mylist_path = os.path.join(temp_directory_path, 'mylist.txt')
with open(mylist_path, "w", encoding="utf-8") as f:
    for glob_list_file in glob_list_files:
        f.write("file '" + glob_list_file + "'\n")
# assert(os.path.exists(mylist_path))

allTS_path = os.path.join(temp_directory_path, 'all.ts')
outputMp4_path = os.path.join(temp_directory_path, 'output.mp4')
outputMp3_path = os.path.join(temp_directory_path, 'output.mp3')
ffmpeg.input(mylist_path, f='concat', safe='0').output(allTS_path, c='copy').run()
# ffmpeg -f concat -safe 0 -i mylist.txt -c copy all.ts
ffmpeg.input(allTS_path).output(outputMp4_path, acodec='copy', bsf='aac_adtstoasc').run()
# ffmpeg -i all.ts -c:v libx264 -c:a copy -bsf:a aac_adtstoasc output.mp4
ffmpeg.input(outputMp4_path).output(outputMp3_path).run()
# ffmpeg -i output.mp4 output.mp3
# rm *.ts output.mp4 mylist.txt

# delete temp files
if DELETE_TEMP_FILES:
    for f in glob(os.path.join(temp_directory_path, "*.ts")):
        os.remove(f)
    os.remove(allTS_path)
    os.remove(outputMp4_path)

print('Epoch needed', epoch_needed)
print('Conversion to datetime', epoch_folder)