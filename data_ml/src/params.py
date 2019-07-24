"""
Global parameters for detection classifier

"""

WINDOW_S = 2.45
INFERENCE_CHUNK_S = 60

SAMPLE_RATE = 20000
N_FFT = 2048
N_MELS = 64
MEL_MIN_FREQ = 200
MEL_MAX_FREQ = 10000

N_GPU = 1
MEAN_FILE = "mean64.txt"
INVSTD_FILE = "invstd64.txt"
MODEL_NAME = "AudioSet_fc_all"
