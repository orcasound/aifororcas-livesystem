import os
import shutil
import random
from pydub import AudioSegment
from librosa import get_duration
from pathlib import Path
from fastai.basic_train import load_learner
from audio.data import AudioConfig, SpectrogramConfig, AudioList
from audio.transform import get_spectro_transforms


def download_newdata(data_path):
    """
    Download the OrcaHello false positives from blob after querying CosmosDB
    """
    pass


def download_original_data(data_path):
    """
    Download the original training data from blob storage
    """
    pass


# Function to pre-process the new audio file
def get_wave_file(wav_file):
    """
    Load a wav file
    """
    return AudioSegment.from_wav(wav_file)


def export_wave_file(audio, begin, end, dest):
    """
    Extract a smaller wav file based start and end duration information
    """
    sub_audio = audio[begin * 1000 : end * 1000]
    sub_audio.export(dest, format="wav")


def extract_segments(audio_path, sample_dict, dest_path, suffix):
    """
    Extract segments given an audio path folder and proposal segments
    """
    # Listing the local audio files
    local_audio_files = str(audio_path) + "/"
    for wav_file in sample_dict.keys():
        audio_file = get_wave_file(local_audio_files + wav_file)
        for begin_time, end_time in sample_dict[wav_file]:
            output_file_name = (
                wav_file.lower().replace(".wav", "")
                + "_"
                + str(begin_time)
                + "_"
                + str(end_time)
                + suffix
                + ".wav"
            )
            output_file_path = dest_path + output_file_name
            export_wave_file(
                audio_file, begin_time, end_time, output_file_path
            )


def pre_process(data_path):
    """
    Convert new audio files into 4s segments, all treated as negatives

    Assumes this dir structure
    ./data_path/
        |-new_data/         (input)
            -somefiles0.wav
            -somefiles1.wav
            ...
        |-new_samples/      (output)
            - asdfas.wav
            - asdfas.wav
            ..

    Args:
        `data_path`: path to data folder
    """

    # Create o/p folder
    local_dir = data_path / "new_samples"
    if os.path.exists(local_dir):
        shutil.rmtree(local_dir)
    os.makedirs(local_dir)

    # iterate over all wav files in new_data folder
    four_sec_dict = {}
    for item in (data_path / "new_data/").glob("*.wav"):
        max_length = get_duration(filename=item) - 1
        four_sec_list = []
        for i in range(int(max_length // 4)):
            four_sec_list.append([i * 4, (i + 1) * 4])

        four_sec_dict[item.name] = four_sec_list

    # get model ready data, split into shorter segments
    extract_segments(
        str(data_path / "new_data/"), four_sec_dict, str(local_dir) + "/", "_Noise"
    )

    # Remove the original data folder
    shutil.rmtree(data_path / "new_data/")


def data_blender(data_path, random_seed=2):
    """
    Function to blend the original data with new data such
    that the total number is the same as the original

    Assumes this dir structure
    ./data_path/
        |-new_samples/
        |-negative/

    After running, the contents of `negative` are overwritten
    `new_samples` is deleted

    Args:
        `data_path`: path to data folder
        `random_seed`: same seed with yield reproducible random samples
    """
    data_path = Path(data_path)
    neg_samples = list((data_path / "negative").glob("*.wav"))
    new_neg_samples = list((data_path / "new_samples").glob("*.wav"))
    n_neg_samples = len(neg_samples)

    # randomly blend neg_samples keeping the same total number as original
    random.seed(random_seed)
    blended_neg_samples = random.sample(
        neg_samples + new_neg_samples,
        k=n_neg_samples
    )

    # copying all blended negative samples
    os.makedirs(data_path / "blended_negative", exist_ok=True)
    for item in blended_neg_samples:
        shutil.copy(src=str(item), dst=str(data_path / "blended_negative"))

    # cleanup directories
    shutil.rmtree(str(data_path / "new_samples"))
    shutil.rmtree(str(data_path / "negative"))
    shutil.move(str(data_path / "blended_negative"), str(data_path / "negative"))


def download_model(data_path):
    """
    Downloaded the last deployed model checkpoint
    """
    pass


def finetune(data_path, model_name, new_model_name):
    """
    Finetune seed model on blended data with a standard fit_one_cycle LR recipe
    """

    # Define AudioConfig needed to create on-the-fly mel spectograms.
    config = AudioConfig(
        standardize=False,
        sg_cfg=SpectrogramConfig(
            f_min=0.0,  # Minimum frequency to display.
            f_max=10000,  # Maximum frequency to display.
            hop_length=256,
            n_fft=2560,  # Number of samples for Fast Fourier Transform (FFT).
            n_mels=256,  # Mel bins.
            pad=0,
            to_db_scale=True,  # Converting to dB scale.
            top_db=100,  # Top decibel sound.
            win_length=None,
            n_mfcc=20,
        ),
    )
    config.duration = 4000  # 4 sec padding or snip.
    config.resample_to = 20000  # Every sample at 20000 frequency.
    config.downmix = True

    # Create DataLoader and put 10% randomly selected data in validation set.
    audios = (
        AudioList.from_folder(data_path, config=config)
        .split_by_rand_pct(0.1, seed=4)
        .label_from_folder()
    )

    # Define transforms to be applied to the data.
    # Frequency masking is enabled to augment data.
    tfms = get_spectro_transforms(mask_time=False, mask_freq=True, roll=False)

    # Create a databunch with batchsize = 64.
    db = audios.transform(tfms).databunch(bs=64)

    # Load model and unfreezing layers to update
    # If cpu argument is set to false, will use GPUs if available
    learner = load_learner(data_path / "models" / model_name, cpu=False)
    learner.unfreeze()

    # Assigning databunch to the model class
    learner.data = db

    # 1-cycle learning (10 epochs and variable learning rate)
    # first described in (https://arxiv.org/abs/1708.07120)
    learner.fit_one_cycle(10, 1e-3)

    # Outputting the new model weights
    learner.export(new_model_name)


if __name__ == "__main__":

    # Defining Path variable
    data_folder = Path("./data/")

    # NOTE: not implemented yet, placeholders
    # download_original_data(data_folder)
    # download_newdata(data_folder)
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

    # pre_process(data_folder)
    # data_blender(data_folder, random_seed=2)
    # download_model(data_folder)  # NOTE: not implemented yet, placeholder
    # finetune(data_folder, "rnd1to10_stg4-rn50.pkl", "rn50-finetuned.pkl")
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
            -model_name(.pkl file)
            -new_model_name(.pkl file)
    """

    # NOTE: not implemented yet, placeholder
    """
    test_model()
    if test is looking good --
        - upload_model(data_folder/'models/'+'new_model_name')
    """
