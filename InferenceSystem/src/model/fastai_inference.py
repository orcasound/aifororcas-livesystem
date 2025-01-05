from fastai.basic_train import load_learner
import pandas as pd
from pydub import AudioSegment
from librosa import get_duration
from pathlib import Path
from numpy import floor
from audio.data import AudioConfig, SpectrogramConfig, AudioList
import os
import shutil
import tempfile


def load_model(mPath, mName="stg2-rn18.pkl"):
    return load_learner(mPath, mName)


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
    Function to exctact segments given a audio path folder and proposal segments
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


class FastAIModel():
    def __init__(self, model_path, model_name="stg2-rn18.pkl", threshold=0.5, min_num_positive_calls_threshold=3):
        self.model = load_model(model_path, model_name)
        self.threshold = threshold
        self.min_num_positive_calls_threshold = min_num_positive_calls_threshold

    def predict(self, wav_file_path):
        '''
        Function which generates local predictions using wavefile
        '''

        # Creates local directory to save 2 second clops
        # local_dir = "./fastai_dir/"
        local_dir = tempfile.mkdtemp()+"/"
        if os.path.exists(local_dir):
            shutil.rmtree(local_dir, ignore_errors=False, onerror=None)
            os.makedirs(local_dir)
        else:
            os.makedirs(local_dir)

        # infer clip length
        max_length = get_duration(filename=wav_file_path)
        print(os.path.basename(wav_file_path))
        print("Length of Audio Clip:{0}".format(max_length))
        #max_length = 60
        # Generating 2 sec proposal with 1 sec hop length
        twoSecList = []
        for i in range(int(floor(max_length)-1)):
            twoSecList.append([i, i+2])

        # Creating a proposal dictionary
        two_sec_dict = {}
        two_sec_dict[Path(wav_file_path).name] = twoSecList

        # Creating 2 sec segments from the defined wavefile using proposals built above.
        # "use_a_real_wavname.wav" will generate -> "use_a_real_wavname_1_3.wav", "use_a_real_wavname_2_4.wav" etc. files in fastai_dir folder
        extract_segments(
            str(Path(wav_file_path).parent),
            two_sec_dict,
            local_dir,
            ""
        )
        
        # Definining Audio config needed to create on the fly mel spectograms
        config = AudioConfig(standardize=False,
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
                                 n_mfcc=20)
                             )
        config.duration = 4000  # 4 sec padding or snip
        config.resample_to = 20000  # Every sample at 20000 frequency
        config.downmix=True

        # Creating a Audio DataLoader
        test_data_folder = Path(local_dir)
        tfms = None
        test = AudioList.from_folder(
            test_data_folder, config=config).split_none().label_empty()
        testdb = test.transform(tfms).databunch(bs=32)

        # Scoring each 2 sec clip
        predictions = []
        pathList = list(pd.Series(test_data_folder.ls()).astype('str'))
        for item in testdb.x:
            predictions.append(self.model.predict(item)[2][1])

        # clean folder
        shutil.rmtree(local_dir)

        # Aggregating predictions

        # Creating a DataFrame
        prediction = pd.DataFrame({'FilePath': pathList, 'confidence': predictions})

        # Converting prediction to float
        prediction['confidence'] = prediction.confidence.astype(float)

        # Extracting Starting time from file name
        prediction['start_time_s'] = prediction.FilePath.apply(lambda x: int(x.split('_')[-2]))

        # Sorting the file based on start_time_s
        prediction = prediction.sort_values(
            ['start_time_s']).reset_index(drop=True)

        # Rolling Window (to average at per second level)
        submission = pd.DataFrame(
                {
                    'wav_filename': Path(wav_file_path).name,
                    'duration_s': 1.0,
                    'confidence': list(prediction.rolling(2)['confidence'].mean().values)
                }
            ).reset_index().rename(columns={'index': 'start_time_s'})

        # Updating first row
        submission.loc[0, 'confidence'] = prediction.confidence[0]

        # Adding lastrow
        lastLine = pd.DataFrame({
            'wav_filename': Path(wav_file_path).name,
            'start_time_s': [submission.start_time_s.max()+1],
            'duration_s': 1.0,
            'confidence': [prediction.confidence[prediction.shape[0]-1]]
            })
        submission = submission.append(lastLine, ignore_index=True)
        submission = submission[['wav_filename', 'start_time_s', 'duration_s', 'confidence']]

        # initialize output JSON
        result_json = {}
        result_json = dict(
            submission=submission,
            local_predictions=list((submission['confidence'] > self.threshold).astype(int)),
            local_confidences=list(submission['confidence'])
        )

        result_json['global_prediction'] = int(sum(result_json["local_predictions"]) >= self.min_num_positive_calls_threshold)
        result_json['global_confidence'] = submission.loc[(submission['confidence'] > self.threshold), 'confidence'].mean()*100
        if pd.isnull(result_json["global_confidence"]):
            result_json["global_confidence"] = 0

        return result_json
