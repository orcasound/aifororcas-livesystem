import pandas as pd
import matplotlib.pyplot as plt
from scipy.io import wavfile

# TODO: Add option to shuffle
# TODO: Extend this class to deal with a list of wav files
class WavMasterFile:
    """
    Given a wav file and sorted annotations (start_times, durations), splits it into windows of postive and negative examples. Object is indexable i.e. wav_master_file[11] returns the relevant window of the wav file and 1/0 label for positive/negative.  
    Internally maintains list of segments and windows used to index into the loaded wav file. 
    """
    def __init__(self,wav_path,start_times,durations,window_duration):
        """
        passed annotations as sorted iterables of start_times, durations 
        parse into negative, positive segments [ segment is a tuple (start_idx,end_idx,label) ]
        split each segment into windows for each
        """
        self.sr, self.wav = wavfile.read(wav_path)
        if self.wav.dtype=="int16":
            self.wav = self.wav.astype('float32') / (2 ** 15)
        self.length = len(self.wav)
        self.start_idxs = [self._s_to_samples(t) for t in start_times]
        self.durations = [ self._s_to_samples(t) for t in durations ]
        self.window_size = self._s_to_samples(window_duration)
        
        # segments on and between annotations
        self.segments = self.segments_from_annotations(self.start_idxs,self.durations,self.window_size)
        
        # split each segment into windows
        self.windows = []
        for segment in self.segments:
            self.windows.extend(self.split_segment_in_windows(segment,self.window_size))
    
    def __len__(self):
        return len(self.windows)
    
    def __getitem__(self,index):
        start_idx, end_idx, label = self.windows[index]
        return self.wav[start_idx:end_idx], label
    
    def _s_to_samples(self,duration):
        return int(duration*self.sr)
    
    def segments_from_annotations(self,start_idxs,durations,window_size):
        # iterate through the annotations in order. if we find a jump
        # add a negative segment. can deal with overlapping segments
        segments, curr_idx = [], 0
        for i in range(len(start_idxs)):
            si, di = start_idxs[i], durations[i]
            if si<self.length: # prevent some errors
                if si-curr_idx>window_size: # add negative segment if gap is large enough 
                    segments.append((curr_idx,si,0)) # use zero for negative
                # add the current annotation 
                segments.append((si,si+di,1)) # use one for positive
                curr_idx = si+di
        if self.length-curr_idx>window_size:
            segments.append((curr_idx,self.length,0)) 
        return segments
    
    def split_segment_in_windows(self,segment,window_size):
        c0, c1, label = segment
        w = window_size
        num_windows = len(range(c0,c1))//w
        if num_windows == 0: return [ segment ]
        else: return [ (c0+i*w,c0+(i+1)*w,label) for i in range(num_windows) ]
    
    def plot_for_debug(self,mode='windows'):
        plot_chunks, yi = [], 0
        if mode=='windows':
            chunks = self.windows
        else: 
            chunks = self.segments
        for c in chunks: # create line segments for each window
            plot_chunks.append((c[0]/self.sr,c[1]/self.sr)) # convert index to time
            plot_chunks.append((yi,yi))
            plot_chunks.append('g' if c[2]==1 else 'r')
            yi += 0.1      
        segplot = plt.plot(*plot_chunks)
        plt.show()
        