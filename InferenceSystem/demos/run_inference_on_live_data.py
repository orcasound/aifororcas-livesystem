from hls_reader import read_files_and_chunk_them
import argparse

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('-modelPath', default='AudioSet_fc_all', type=str, required=True)
    parser.add_argument('-clipInSeconds', default=60, type=int, required=False)
    parser.add_argument('-sleepInSeconds', default=50, type=int, required=False)
    parser.add_argument('-localPredictionThreshold', default=0.5, type=float, required=False)
    parser.add_argument('-spectrogramBufferLength', default=10, type=int, required=False, help="Number of spectrograms to save before you start overwriting")
    args = parser.parse_args()
    
    read_files_and_chunk_them(args)

