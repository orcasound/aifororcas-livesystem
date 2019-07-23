import os
import numpy as np
from scipy.io import wavfile
from collections import defaultdict
from src.dataloader import AudioFileWindower 
from pathlib import Path
# inputs: model, mean invstd, input_wav, output_chunk_dir, output_json_dir

# iterate through windows and save audio chunks
def write_chunks(audio_file_windower,chunk_duration,output_dir_path):
    audio_file_windower.get_mode = 'audio'
    curr_chunk, curr_chunk_duration = [], 0.
    file_chunk_counts = defaultdict(int)
    for i in range(len(audio_file_windower)):
        audio_window, _ = audio_file_windower[i]
        _, _, _, af = audio_file_windower.windows[i]
        curr_chunk_duration += audio_file_windower.window_s
        curr_chunk.append(audio_window)
        # if exceeds chunk_duration, write and reset
        if curr_chunk_duration > chunk_duration:
            # create a filename for it 
            postfix = format(file_chunk_counts[af.name]+1,'04x')
            output_file_path = output_dir_path / (Path(af.name).stem+postfix+'.wav')
            # write out the chunk and reset
            print("Writing out chunk:",output_file_path.name)
            wavfile.write(output_file_path,af.sr,np.concatenate(curr_chunk))
            curr_chunk, curr_chunk_duration = [], 0.
            file_chunk_counts[af.name] += 1

if __name__ == "__main__":
    wavmaster_path = Path("../data/wavmaster")
    output_dir_path = Path("../data/wavmaster_chunked")
    wav_file_paths = [ wavmaster_path/p for p in os.listdir(wavmaster_path) ]
    windower = AudioFileWindower(wav_file_paths[:2])
    write_chunks(windower,90,output_dir_path)