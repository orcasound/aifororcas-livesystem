import os
from math import ceil
from pathlib import Path

import librosa
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.io import wavfile
from torch.utils.data import Dataset
from tqdm import tqdm

from . import params


def s_to_samples(duration,sr):
    return int(duration*sr)

class AudioFile:
    """
    Attributes:
        sr (int)
        nsamples (int)
        duration (float)
        audio (float32 array)
        name (str)
    """
    def __init__(self,file_path,target_sr):
        file_path = Path(file_path) 
        self.name = file_path.name

        if file_path.suffix == '.wav':
            sr, audio = wavfile.read(file_path)
            if audio.dtype=="int16":
                audio = audio.astype('float32') / (2 ** 15)
            elif audio.dtype=="float32":
                pass
            else:
                raise Exception("Error, wav format {} not supported for {}".format(audio.dtype,self.name)) 
            # if multichannel wav recordings, use the first channel
            if len(audio.shape)>1:
                audio = audio[:,0]

            if sr != target_sr: # convert to a common sampling rate
                print("Warning: Resampling file {} with SR: {}, dtype: {}".format(
                    self.name, target_sr, audio.dtype)
                )
                self.audio_original = audio
                self.sr_original = sr
                og_directory = Path(file_path).parent / "original_{:.1f}_kHz".format(sr/1000.0)
                os.makedirs(og_directory, exist_ok=True)
                wavfile.write(og_directory / self.name, self.sr_original, self.audio_original)
                audio = librosa.core.resample(audio, sr, target_sr) 
                wavfile.write(file_path, target_sr, audio)
                print("Overwritten file at {} and copied original to {}".format(file_path, og_directory))
            else:
                self.audio_original = audio
                self.sr_original = target_sr
            self.sr, self.audio = target_sr, audio
        self.nsamples = len(self.audio)
        self.duration = self.nsamples/self.sr
    
    def extend(self,target_duration_s):
        target_nsamples = s_to_samples(target_duration_s,self.sr)
        if target_nsamples > self.nsamples:
            audio_tiled = np.tile(self.audio,ceil(target_nsamples/self.nsamples))
            self.audio = audio_tiled
            self.nsamples = len(self.audio)
            self.duration = self.nsamples/self.sr
    
    def get_window(self,start_idx,end_idx,mode='mel_spec'):
        audio_window = self.audio[start_idx:end_idx]
        if mode=='audio':
            return audio_window 
        elif mode=='audio_orig_sr':
            start_idx = int(start_idx*self.sr_original/self.sr)
            end_idx = int(end_idx*self.sr_original/self.sr)
            return self.audio_original[start_idx:end_idx]
        elif mode=='spec':
            spec = np.abs(librosa.core.stft(
                audio_window,
                n_fft=params.N_FFT,
                hop_length=int(params.HOP_S*self.sr)
                )) # ok with defaults n_fft=2048 
            return np.log(spec).T # dimension: T x F
        elif mode=='mel_spec':
            spec = np.abs(librosa.core.stft(
                audio_window,
                n_fft=params.N_FFT,
                hop_length=int(params.HOP_S*self.sr)
                )) # ok with defaults n_fft=2048
            # roughly trying out some params based on https://seaworld.org/animals/all-about/killer-whale/communication/
            mel_fbank = librosa.filters.mel(
                self.sr,
                n_fft=params.N_FFT,
                n_mels=params.N_MELS,
                fmin=params.MEL_MIN_FREQ,
                fmax=params.MEL_MAX_FREQ
            )
            mel_spec = np.dot(mel_fbank,spec)
            return np.log(mel_spec).T # dimension: T x F


class AudioFileDataset(Dataset):
    """
    Given a tsv with (wav_file,start_time,duration) loads audio and indexes it into windows.
    AudioFileDataset[index] returns (window,label), where window can be audio, spectrogram, or mel_spectrum depending on get_mode.  

    Internally indexes AudioFiles and maintains list of segments and windows used to index into them. Also extends audio files < min_window_s by repeating them. 
    """
    def __init__(self, wav_dir, tsv_file, 
        min_window_s=params.WINDOW_S, max_window_s=params.WINDOW_S, hop_s=0.0,
        mean=None, invstd=None, sr=params.SAMPLE_RATE, 
        get_mode='mel_spec', transform=None, jitter=False, random_seed=42):
        # wav_dir, tsv_file, max_window_s
        """
        load all wavfiles into memory (data is not too large so can get away with this, else use memmap option while reading wavfiles)
        """
        self.df = pd.read_csv(tsv_file,sep='\t')
        self.max_window_s = max_window_s
        self.min_window_s = min_window_s
        if hop_s == 0.0:
            self.hop_s = self.min_window_s
        else:
            self.hop_s = hop_s
        self.transform = transform
        self.jitter = jitter
        self.random_seed = random_seed
        np.random.seed(self.random_seed)
        assert get_mode in ['audio','spec','mel_spec', 'audio_orig_sr']
        self.sr, self.get_mode = sr, get_mode

        self.audio_files, self.segments, self.windows = {}, [], []
        wav_iterator = tqdm(self.df.wav_filename.unique())
        for wav_filename in wav_iterator:
            wav_iterator.set_description(wav_filename)
            wav_df = self.df[self.df['wav_filename']==wav_filename]
            wav_path = Path(wav_dir)/wav_filename
            audio_file = AudioFile(wav_path,self.sr)
            audio_file.extend(self.min_window_s)
            start_times, durations = wav_df['start_time_s'], wav_df['duration_s']
            wav_segments, wav_windows = self.index_audio_file(
                audio_file,start_times,durations,
                self.min_window_s, self.max_window_s,
                hop_s=self.hop_s
                )
            self.segments.extend(wav_segments)
            self.windows.extend(wav_windows)
            self.audio_files[wav_filename] = audio_file 

        # if mean and invstd were not provided, calculate them
        if os.path.exists(mean) and os.path.exists(invstd):
            self.mean, self.invstd = np.loadtxt(mean), np.loadtxt(invstd)
            print("Loaded mean and invstd from:",mean,invstd)
        else:
            # calculate the mean and invstd from data 
            self.mean, self.invstd = self.calculate_mean_and_invstd()
            np.savetxt(mean, self.mean)
            np.savetxt(invstd, self.invstd)
    
    def calculate_mean_and_invstd(self):
        mean = np.zeros(params.N_MELS)
        num_windows = len(self.windows)
        for i in range(num_windows):
            start_idx, end_idx, label, audio_file = self.windows[i]
            data = audio_file.get_window(start_idx,end_idx,self.get_mode)
            mean += data.mean(axis=0)

        mean = mean/num_windows

        variance = np.zeros(params.N_MELS)
        for i in range(num_windows):
            start_idx, end_idx, label, audio_file = self.windows[i]
            data = audio_file.get_window(start_idx,end_idx,self.get_mode)
            variance += ((data-mean)**2).mean(axis=0)
        variance /= num_windows
        invstd = 1/np.sqrt(variance)

        return mean, invstd
        
    def index_audio_file(self,audio_file,start_times,durations,min_window_s,max_window_s, hop_s=0.0):
        """
        Given an audio_file and sorted annotations (start_times, durations), creates sequential segments and windows of postive and negative examples.
        First, create sequential segments of min_window_s
        [ segment: a tuple (start_idx,end_idx,label,audio_file) ]
        Then split each segment into windows with max_window_s 

        Arguments:
            audio_file (AudioFile)
            start_times (iterable)
            durations (iterable)
            min_window_s (float)
            max_window_s (float)
        Returns:
            segments (list)
            windows (list)
        """
        
        # segments on and between annotations
        segments = self.segments_from_annotations(
            audio_file, start_times, durations, min_window_s
            )
        # split each segment into windows
        windows = []
        for segment in segments:
            windows.extend(
                self.split_segment_in_windows(audio_file, segment, max_window_s, hop_s)
                )
        
        return segments, windows
    
    def __len__(self):
        return len(self.windows)
    
    def __getitem__(self,index):
        start_idx, end_idx, label, audio_file = self.windows[index]
        if self.jitter:
            # perturb windows so each audio file is processed slightly differently each epoch
            hop_idx = s_to_samples(self.hop_s, self.sr)
            perturb_min = min(start_idx, hop_idx)
            perturb_max = min(audio_file.nsamples - end_idx, hop_idx)
            perturb_idx = np.random.randint(-perturb_min, perturb_max)
            start_idx += perturb_idx
            end_idx += perturb_idx
        data = audio_file.get_window(start_idx, end_idx, self.get_mode)
        if (self.mean is not None) and (self.invstd is not None) and ('audio' not in self.get_mode):
            data -= self.mean
            data *= self.invstd 
        if self.transform is not None:
            data = self.transform(data)
        return data, label
    
    def segments_from_annotations(self,audio_file,start_times,durations,min_window_s):
        sr = audio_file.sr
        start_idxs = [s_to_samples(t,sr) for t in start_times]
        duration_idxs = [s_to_samples(t,sr) for t in durations ]
        min_window_idx = s_to_samples(min_window_s,sr)

        # iterate through the annotations in order. if we find a jump
        # add a negative segment. can deal with overlapping segments
        segments, curr_idx, nsamples = [], 0, audio_file.nsamples 
        for i in range(len(start_idxs)):
            si, di = start_idxs[i], duration_idxs[i]
            ei = si+di # end of segment

            if si < nsamples: # prevent some errors

                if (si-curr_idx) >= min_window_idx: # add negative segment if gap is large enough 
                    segments.append((curr_idx,si,0,audio_file)) # use zero for negative

                # add annotation directly if it's a minimum size 
                if (di >= min_window_idx): 
                    segments.append((si,min(ei,nsamples),1,audio_file)) # use one for positive
                # extend annotation if possible (segment from master tape)
                elif (si+min_window_idx) < nsamples: 
                    ei = si+min_window_idx
                    segments.append((si,ei,1,audio_file)) # use one for positive

                curr_idx = ei

        if (nsamples-curr_idx) >= min_window_idx: # adding empty segment at end
            segments.append((curr_idx,nsamples,0,audio_file)) 
        return segments
    
    def split_segment_in_windows(self,audio_file,segment,max_window_s, hop_s=0.0):
        """
        Splits a segment (start, end, label, wav) into non-overlapping chunks of window_size

        An example illustration with (len = 7, window = 3, hop = 1)
        Samples:    - - - - - - -
        Indexes:    0 1 2 3 4 5 6
        Windows:    * * * * *          total = 5
        Formula:    (len - window)/hop + 1
        """
        si, ei, label, audio_file = segment  # si, ei are sample indexes of the segment
        w = s_to_samples(max_window_s, audio_file.sr) 
        # if no hop specified, defaults to hop=window size i.e. non-overlapping windows
        hop = s_to_samples(hop_s, audio_file.sr) if hop_s > 0.0 else w
        num_windows = (len(range(si,ei)) - w) // hop + 1

        if num_windows == 0:
            return [segment]  # smaller than max
        else:
            windows_list = []
            for i in range(num_windows):
                start = si + i*hop
                end = start + w
                current_window = (start, end, label, audio_file)
                #TODO@Akash: support some random jitter when splitting into windows
                windows_list.append(current_window)
            return windows_list
    
    def plot_for_debug(self,audio_fname,mode='windows'):
        plot_chunks, yi, sr = [], 0, self.audio_files[audio_fname].sr
        if mode=='windows':
            chunks = [ w for w in self.windows if w[-1].name == audio_fname ]
        else: 
            chunks = [ s for s in self.segments if s[-1].name == audio_fname ]
        for c in chunks: # create line segments for each window
            plot_chunks.append((c[0]/sr,c[1]/sr)) # convert index to time
            plot_chunks.append((yi,yi))
            plot_chunks.append('g' if c[2]==1 else 'r')
            yi += 0.1      
        _ = plt.plot(*plot_chunks)
        plt.show()


class AudioFileWindower(AudioFileDataset):
    def __init__(self,
        audio_file_paths,window_s=params.WINDOW_S, hop_s=0.0, mean=None,invstd=None,sr=params.SAMPLE_RATE,get_mode='mel_spec',transform=None):
        """
        load all wavfiles into memory (data is not too large so can get away with this, else use memmap option while reading wavfiles)
        """
        # 
        self.audio_file_paths = [ Path(p) for p in audio_file_paths ]
        self.window_s = window_s
        self.transform = transform
        self.jitter = False
        if (mean is not None) and (invstd is not None):
            self.mean, self.invstd = np.loadtxt(mean), np.loadtxt(invstd)
            print("Loaded mean and invstd from:",mean,invstd)
        else:
            self.mean, self.invstd = None, None
        assert get_mode in ['audio','spec','mel_spec']
        self.sr, self.get_mode = sr, get_mode
        self.audio_files, self.segments, self.windows = {}, [], []
        for audio_file_path in self.audio_file_paths:
            print("Loading file:",audio_file_path.name)
            try:
                audio_file = AudioFile(audio_file_path,self.sr)
                audio_file.extend(self.window_s)
                start_times, durations = [0.], [audio_file.duration]
                wav_segments, wav_windows = self.index_audio_file(
                    audio_file,start_times,durations,
                    self.window_s, self.window_s, hop_s
                    )
                self.segments.extend(wav_segments)
                self.windows.extend(wav_windows)
                self.audio_files[audio_file_path.name] = audio_file 
            except Exception as e:
                print("Error with file:",audio_file_path.name,e)

def debug_error_with_indexing():
    dataset = AudioFileDataset("../train_data/wav","../train_data/train.tsv",2,2)
    spec_shapes = []
    error_idxs = []
    for i in range(len(dataset)):
        spec_shapes.append(dataset[i][0].shape)
        if spec_shapes[i] != (79,80):
            error_idxs.append(i)
    return spec_shapes, error_idxs
        
if __name__ == "__main__":
    debug_error_with_indexing()