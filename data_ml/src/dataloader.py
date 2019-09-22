import librosa
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from pathlib import Path
from math import ceil
from torch.utils.data import Dataset
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
                print("Warning: Overwriting file {} with SR: {}, dtype: {}".format(file_path,target_sr, audio.dtype))
                audio = librosa.core.resample(audio,sr,target_sr) 
                wavfile.write(file_path,target_sr,audio)
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
        elif mode=='spec':
            spec = np.abs(librosa.core.stft(audio_window,n_fft=params.N_FFT)) # ok with defaults n_fft=2048, hop 1/4th
            return np.log(spec).T # dimension: T x F
        elif mode=='mel_spec':
            spec = np.abs(librosa.core.stft(audio_window,n_fft=params.N_FFT)) # ok with defaults n_fft=2048, hop 1/4th
            # roughly trying out some params based on https://seaworld.org/animals/all-about/killer-whale/communication/
            mel_fbank = librosa.filters.mel(
                self.sr,n_fft=params.N_FFT,n_mels=params.N_MELS,
                fmin=params.MEL_MIN_FREQ,fmax=params.MEL_MAX_FREQ)
            mel_spec = np.dot(mel_fbank,spec)
            return np.log(mel_spec).T # dimension: T x F


class AudioFileDataset(Dataset):
    """
    Given a tsv with (wav_file,start_time,duration) loads audio and indexes it into windows.
    AudioFileDataset[index] returns (window,label), where window can be audio, spectrogram, or mel_spectrum depending on get_mode.  

    Internally indexes AudioFiles and maintains list of segments and windows used to index into them. Also extends audio files < min_window_s by repeating them. 
    """
    def __init__(
        self,wav_dir,tsv_file,min_window_s=params.WINDOW_S,max_window_s=params.WINDOW_S,
        mean=None,invstd=None,sr=params.SAMPLE_RATE,get_mode='mel_spec',transform=None):
        # wav_dir, tsv_file, max_window_s
        """
        load all wavfiles into memory (data is not too large so can get away with this, else use memmap option while reading wavfiles)
        """
        self.df = pd.read_csv(tsv_file,sep='\t')
        self.max_window_s = max_window_s
        self.min_window_s = min_window_s
        self.transform = transform
        if (mean is not None) and (invstd is not None):
            self.mean, self.invstd = np.loadtxt(mean), np.loadtxt(invstd)
            print("Loaded mean and invstd from:",mean,invstd)
        else:
            self.mean, self.invstd = None, None
        assert get_mode in ['audio','spec','mel_spec']
        self.sr, self.get_mode = sr, get_mode
        self.audio_files, self.segments, self.windows = {}, [], []
        for wav_filename in self.df.wav_filename.unique():
            wav_df = self.df[self.df['wav_filename']==wav_filename]
            wav_path = Path(wav_dir)/wav_filename
            print("Loading file:",wav_filename)
            audio_file = AudioFile(wav_path,self.sr)
            audio_file.extend(self.min_window_s)
            start_times, durations = wav_df['start_time_s'], wav_df['duration_s']
            wav_segments, wav_windows = self.index_audio_file(
                audio_file,start_times,durations,
                self.min_window_s, self.max_window_s
                )
            self.segments.extend(wav_segments)
            self.windows.extend(wav_windows)
            self.audio_files[wav_filename] = audio_file 
    
    def index_audio_file(self,audio_file,start_times,durations,min_window_s,max_window_s):
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
            audio_file,start_times,durations,min_window_s
            )
        # split each segment into windows
        windows = []
        for segment in segments:
            windows.extend(self.split_segment_in_windows(audio_file,segment,max_window_s))
        
        return segments, windows
    
    def __len__(self):
        return len(self.windows)
    
    def __getitem__(self,index):
        start_idx, end_idx, label, audio_file = self.windows[index]
        data = audio_file.get_window(start_idx,end_idx,self.get_mode)
        if (self.mean is not None) and (self.invstd is not None) and self.get_mode != 'audio':
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
    
    def split_segment_in_windows(self,audio_file,segment,max_window_s):
        # splits a segment (start,end,label,wav) into non-overlapping chunks of window_size
        w = s_to_samples(max_window_s,audio_file.sr) 
        si, ei, label, audio_file = segment
        num_windows = len(range(si,ei))//w
        if num_windows == 0: return [ segment ] # smaller than max
        else: return [ (si+i*w,si+(i+1)*w,label,audio_file) for i in range(num_windows) ]
    
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
        audio_file_paths,window_s=params.WINDOW_S,mean=None,invstd=None,sr=params.SAMPLE_RATE,get_mode='mel_spec',transform=None):
        """
        load all wavfiles into memory (data is not too large so can get away with this, else use memmap option while reading wavfiles)
        """
        # 
        self.audio_file_paths = [ Path(p) for p in audio_file_paths ]
        self.window_s = window_s
        self.transform = transform
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
                    self.window_s, self.window_s
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