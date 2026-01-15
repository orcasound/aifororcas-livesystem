import argparse
import multiprocessing
import os
import pdb
import sys
from glob import glob
from pathlib import Path

import pandas as pd
from tqdm import tqdm

SRC_DIR = Path(__file__).parent.parent / "src"
print(f"Adding {SRC_DIR} to sys.path")
sys.path.append(str(SRC_DIR))

from model.fastai_inference import FastAIModel
from model.podcast_inference import OrcaDetectionModel


class ForkedPdb(pdb.Pdb):
    """A Pdb subclass that may be used
    from a forked multiprocessing child
    https://github.com/williamFalcon/forked-pdb
    """
    def interaction(self, *args, **kwargs):
        _stdin = sys.stdin
        try:
            sys.stdin = open('/dev/stdin')
            pdb.Pdb.interaction(self, *args, **kwargs)
        finally:
            sys.stdin = _stdin

def split_list(l, n): 
    """Yield successive n-sized chunks from l."""
    # looping till length l 
    for i in range(0, len(l), n):  
        yield l[i:i + n] 
  

class ModelEvaluator:
    """
    Simple harness for multiprocessing-friendly test set evaluation. Subclass and override 
    `load_model` and `process_wav_list` methods appropriately if adding a different 
    unsupported model_type. 

    The only requirement is that: 
        - `process_wav_list` should call load_model() on each run
        - `process_wav_list` should write (for each wav file) a
           submission-format TSV file to self.results_dir. 
    """
    def __init__(self, args):
        self.wav_dir = Path(args.testset_wav_dir)
        self.model_path = Path(args.model_path)
        self.results_dir = Path(args.results_dir)
        os.makedirs(self.results_dir, exist_ok=True)
        self.model_type = args.model_type
    
    def process_wav_list(self, wav_list):
        """
        """
        # NOTE: Pytorch model needs to be loaded separately in each process call, due to https://github.com/pytorch/pytorch/issues/35472 
        model = self.load_model()

        for i, wav_file in enumerate(wav_list):
            print("Processing {}/{} in process {}".format(i, len(wav_list)-1, os.getpid()))
            results_tsv = self.results_dir / "{}.tsv".format(Path(wav_file).stem)
            if not os.path.isfile(results_tsv):
                res_df = model.predict(wav_file)['submission']
                res_df.to_csv(results_tsv, sep='\t', index=False)
            else:
                print("Predictions already generated for {}, re-using".format(results_tsv))
    
    def load_model(self):
        """
        Initializes an object that has an appropriate predict() method. 
        """
        model_local_threshold = 0.0
        model_global_threshold = 1

        # load and generate submission file for FastAI-ResNet model
        assert self.model_type in ['Baseline-AudioSet', 'FastAI'], \
            "Unsupported model_type! Please double check args."
        if self.model_type == 'Baseline-AudioSet':
            model = OrcaDetectionModel(
                model_path=self.model_path, threshold=model_local_threshold, 
                min_num_positive_calls_threshold=model_global_threshold, 
                hop_s=1.0, rolling_avg=True
            )
        elif self.model_type == 'FastAI':
            model_dirpath = self.model_path.parent
            model_name = self.model_path.name
            model = FastAIModel(
                model_path=model_dirpath, model_name=model_name, threshold=model_local_threshold, 
                min_num_positive_calls_threshold=model_global_threshold
            )
        
        return model
        

if __name__ == "__main__":

    description = \
        """
        Evaluation script for quick multi-process inference on a test set. 
        Currently supports `model_type`: (Baseline-AudioSet/FastAI). 
        Refer to the ModelEvaluator class to easily extend to other models. 
        A submission-format TSV file is written to `results_dir` along with
        individual per-wav TSV files. 
        """

    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('testset_wav_dir', type=str)
    parser.add_argument('model_path', type=str)
    parser.add_argument('results_dir', type=str)
    parser.add_argument('--model_type', type=str, required=True, help="One of (Baseline-AudioSet/FastAI)")
    parser.add_argument('--num_processes', type=int, default=2, help="Multiprocessing pool size")
    args = parser.parse_args()

    wavfiles = glob("{}/*.wav".format(args.testset_wav_dir))
    # wavfiles = wavfiles[1:11]  # Debugging 
    wav_list_for_evaluator = split_list(wavfiles, len(wavfiles)//args.num_processes)

    model_evaluator = ModelEvaluator(args)

    # multiprocess inference 
    pool = multiprocessing.Pool(processes = args.num_processes)
    pool.map(model_evaluator.process_wav_list, wav_list_for_evaluator)
    pool.close()

    # merge results into single submission file
    submission_file = Path(args.results_dir)/"{}.tsv".format(Path(args.results_dir).name)
    if submission_file.is_file():
        os.remove(submission_file)
    submission = pd.concat(
                [pd.read_csv(t, sep='\t') for t in glob("{}/*.tsv".format(args.results_dir))]
            )
    submission.to_csv(submission_file, sep='\t', index=False)
    print("Written a submission file for the test set to:", submission_file)
