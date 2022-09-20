from fastai.basic_train import load_learner
import pandas as pd
import numpy as np

from pydub import AudioSegment
from librosa import get_duration
from pathlib import Path
from numpy import floor
from audio.data import AudioConfig, SpectrogramConfig, AudioList
from audio.transform import get_spectro_transforms
import os
import shutil
import matplotlib.pyplot as plt

                          
# Defining Path variable
data_folder = Path("./data/")



def download_newdata():
    pass
# download_newdata() : TBA

"""
Resulting Folder Structure - 
./data/   
    |-new_data/
        -somefiles0.wav
        -somefiles1.wav
        ...
"""


def download_original_data():
    """
    Function to download the original training data from blob storage
    """
    pass

## download_original_data() : TBA

"""
Resulting Folder Structure - 
./data/   
    |-positive/
        - xyz1.wav
        - xyz2.wav
        ...
    |-negative/
        - abc1.wav
        - abc2.wav
        ...
    |-new_data/
        -somefiles0.wav
        -somefiles1.wav
        ...
"""

# Function to pre-process the new audio file
def get_wave_file(wav_file):
    '''
    Function to load a wav file
    '''
    return AudioSegment.from_wav(wav_file)


def export_wave_file(audio, begin, end, dest):
    '''
    Function to extract a smaller wav file based start and end duration information
    '''
    sub_audio = audio[begin * 1000:end * 1000]
    sub_audio.export(dest, format="wav")


def extract_segments(audioPath, sampleDict, destnPath, suffix):
    '''
    Function to extract segments given an audio path folder and proposal segments
    '''
    # Listing the local audio files
    local_audio_files = str(audioPath) + '/'
    for wav_file in sampleDict.keys():
        audio_file = get_wave_file(local_audio_files + wav_file)
        for begin_time, end_time in sampleDict[wav_file]:
            output_file_name = wav_file.lower().replace(
                '.wav', '') + '_' + str(begin_time) + '_' + str(
                    end_time) + suffix + '.wav'
            output_file_path = destnPath + output_file_name
            export_wave_file(audio_file, begin_time,
                             end_time, output_file_path)

def pre_process(dataPath=data_folder):
    """
    Function to convert new audio file containing False Negative into model-ready stream
    Input -
    dataPath: path to data folder
    Output -
    Will automatically put processed files in the 'filePath/new_samples/' folder
    """

    ## Create o/p folder 
    local_dir = dataPath/"new_samples"
    if os.path.exists(local_dir):
        shutil.rmtree(local_dir)
    os.makedirs(local_dir)

    ## iterate over all wav files in new_data folder
    four_sec_dict = {}
    for item in (dataPath/"new_data/").glob("*.wav"):
        max_length = get_duration(filename=item) - 1 
        fourSecList = []
        for i in range(int(max_length//4)):
            fourSecList.append([i*4, (i+1)*4])

        four_sec_dict[item.name] = fourSecList
    
    ## get model ready data
    extract_segments(
            str(dataPath/"new_data/"),
            four_sec_dict,
            str(local_dir)+'/',
            "_Noise"
        )
    
    ## Remove the original data folder
    shutil.rmtree(dataPath/"new_data/")

pre_process(Path("./data/"))

"""
Resulting Folder Structure - 
./data/   
    |-positive/
        - xyz1.wav
        - xyz2.wav
        ...
    |-negative/
        - abc1.wav
        - abc2.wav
        ..
    |-new_samples/
        - asdfas.wav
        - asdfas.wav
        ..
"""

def data_blender(dataPath):
    """
    Function to blend the original data with new data
    
    """
    pos_samples  = len((dataPath/'positive').ls())
    neg_samples  = len((dataPath/'negative').ls())
    new_neg_samples  = len((dataPath/'new_samples').ls())
    total_neg_samples = neg_samples + new_neg_samples
    
    neg_img_list = pd.Series((dataPath/'negative').ls() + (dataPath/'new_samples').ls())
    
    ## Randomly selecting neg_samples for overall list
    new_neg_img_list = neg_img_list.sample(n=neg_samples, replace=False).values
    
    ## Get image name
    new_neg_img_list = [str(item).split('/')[-1] for item in new_neg_img_list]
    
    ## copying all data from new samples to negative
    for item in (dataPath/'new_samples').ls():
        shutil.move(src= str(item), dst = dataPath/'negative', copy_function = shutil.copy)
        
    ## removing new samples directory
    shutil.rmtree(dataPath/'new_samples')
        
    for item in (dataPath/'negative').ls():
        if str(item).split('/')[-1] not in new_neg_img_list:
            os.remove(item)

data_blender()
"""
Resulting Folder Structure - 
./data/   
    |-positive/
        - xyz1.wav
        - xyz2.wav
        ...
    |-negative/
        - abc1.wav
        - abc2.wav
"""

def download_model():
    pass
# download_model()


"""
Folder Structure - 
./data/   
    |-positive/
        - xyz1.wav
        - xyz2.wav
        ...
    |-negative/
        - abc1.wav
        - abc2.wav
        ...
    |-models/
        -modelName.pkl
"""

def finetune(dataPath=data_folder, modelName="rnd1to10_stg4-rn50.pkl", newModelName="rnd1to10_stg4-rn50.pkl"):
    """
    Function to do finetuning of the model
    """

    # Define AudioConfig needed to create on-the-fly mel spectograms.
    config = AudioConfig(
        standardize=False,
        sg_cfg=SpectrogramConfig(
            f_min=0.0,  # Minimum frequency to Display
            f_max=10000,  # Maximum Frequency to Display
            hop_length=256,
            n_fft=2560,  # Number of Samples for Fourier
            n_mels=256,  # Mel bins
            pad=0,
            to_db_scale=True,  # Converting to DB sclae
            top_db=100,  # Top decible sound
            win_length=None,
            n_mfcc=20,
        ),
    )
    config.duration = 4000  # 4 sec padding or snip
    config.resample_to = 20000  # Every sample at 20000 frequency
    config.downmix = True

    # Create DataLoader and put 10% of randomly selected data in the validation set.
    audios = (
        AudioList.from_folder(data_folder, config=config)
        .split_by_rand_pct(0.1, seed=4)
        .label_from_folder()
    )

    ## Defining Transformation
    ## Frequency masking:ON
    tfms = get_spectro_transforms(mask_time=False, mask_freq=True, roll=False) 

    ## Creating a databunch
    db = audios.transform(tfms).databunch(bs=64)


    ## Load model and unfreezing layers to update
    model = load_learner(data_folder / "models", modelName)
    learn.unfreeze()

    ## Assigning databunch to the model class
    learn.data = db


    ## 1-cycle learning (10 epochs and variable learning rate)
    learn.fit_one_cycle(10, 1e-3)

    ## Outputting the new model weights
    learn.export(newModelName)

"""
Folder Structure - 
./data/   
    |-positive/
        - xyz1.wav
        - xyz2.wav
        ...
    |-negative/
        - abc1.wav
        - abc2.wav
        ...
    |-models/
        -modelName(.pkl file)
        -newModelName(.pkl file)
"""

'''
Still need to write the function
test_model()
if test is looking good --
    - upload_model(data_folder/'models/'+'newModelName')
'''