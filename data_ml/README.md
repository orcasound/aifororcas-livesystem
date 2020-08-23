# Setup instructions 🎱 🐋

1. [Windows] Get [pyenv-win](https://github.com/pyenv-win/pyenv-win) to manage python versions:
    1. `git clone https://github.com/pyenv-win/pyenv-win.git %USERPROFILE%/.pyenv` 
    2. Add the following to your shell PATH `%USERPROFILE%\.pyenv\pyenv-win\bin`, `%USERPROFILE%\.pyenv\pyenv-win\shims` 

2. [Mac] Get [pyenv](https://github.com/pyenv/pyenv) to manage python versions:
	1. Use homebrew and run `brew update && brew install pyenv`
	2. Follow from step 3 onwards [here](https://github.com/pyenv/pyenv#basic-github-checkout). This essentially adds the `pyenv init` command to your shell on startup 
	3. FYI this is a [commands reference](https://github.com/pyenv/pyenv/blob/master/COMMANDS.md)

3. [Common] Install and maintain the right Python version (3.6.8) 
    1. Run `pyenv --version` to check installation 
    2. Run `pyenv rehash` from your home directory, install python 3.6.8 with `pyenv install -l 3.6.8` (use 3.6.8-amd64 on Windows if relevant) and run `pyenv rehash` again 
    3. Cd to `/PodCast` and set a local python version `pyenv local 3.6.8` (or 3.6.8-amd64). This saves a `.python-version` file that tells pyenv what to use in this dir 
    4. Type `python --version` and check you're using the right one

(feel free to skip 1, 2, 3 if you prefer to use your own Python setup and are familiar with many of this)

4. Create a [virtual environment](https://docs.python.org/3.6/library/venv.html) to isolate and install package dependencies 
    1. In your working directory, run `python -m venv podcast-venv`. This creates a directory `podcast-venv` with relevant files/scripts. 
	2. On Mac, activate this environment with `source podcast-venv/bin/activate` and when you're done, `deactivate`
	   On Windows, activate with `.\podcast-venv\Scripts\activate.bat` and `.\podcast-venv\Scripts\deactivate.bat` when done
    3. In an active environment, cd to `/data_ml` and run `python -m pip install --upgrade pip && pip install -r requirements.txt` 

# Directory structure:

```
PodCast
	- data_ml 						(current directory)
		- src						(module library)
	- models 						
		- pytorch_vggish.pth		(pretrained AudioSet model)
		- PodCastRound2V1.0			(selected trained model)
	- runs 
		- [MODELNAME]				(model checkpoints etc. while training)
	- [TRAIN_DATA_ARCHIVE_DIR]			
		- wav						(contains raw wav files)
		- train.tsv					
		- dev.tsv 
	- [TEST_DATA_ARCHIVE_DIR]		(similar as above. only use for final model evaluation, not tuning)
```

See documentation at [DataArchives](https://github.com/orcasound/orcadata/wiki/Pod.Cast-data-archive) for details on how to access and read datasets in a standard form. 

# Approach

The implementation here focuses on *detection of orca calls*, that are in the audible range, hence fun to listen to and annotate :)
For now, we simply fine-tune the fully-connected and classification head of the [AudioSet model](https://github.com/tensorflow/models/tree/master/research/audioset), specifically a [PyTorch port of the model/weights](https://github.com/tcvrick/audioset-vggish-tensorflow-to-pytorch). The model is generating *local predictions* on a fixed window size of ~2.45s. Sampling and aggregation strategies for more *global detection* at minute/hourly/day-wise time scale would be a welcome contribution. This is heplful when deploying a real-time detection pipeline, or processing 2-3 months of historical data from different hydrophone nodes. 

> 1. Labelled data in matched live conditions (hydrophone audio capture pipeline, location, acoustic environment) is limited (<5k examples). We bootstrap with some scraped open data from WHOIS (see [DataArchives](https://github.com/orcasound/orcadata/wiki/Pod.Cast-data-archive)) 
> 2. Given limited domain data, and the need for robustness to different acoustic conditions (hydrophone nodes, SNR, noise sources and disturbances) in live conditions, we believe transfer learning would be essential to obtaining the most reliable performance  
> 3. Data augmentation in the style of [SpecAug](https://arxiv.org/pdf/1904.08779.pdf) is also implemented, that acts as a helpful form of regularization 
> 4. The Pod.Cast website aims to generate labelled data in live conditions, with candidates for annotation created in an [active-learning-like](https://en.wikipedia.org/wiki/Active_learning_(machine_learning)) fashion. The goal is to generate more relevant labelled data through multiple rounds of the above feedback loop bootstrapped by the classifier 


# Code examples 

Pardon the brevity here, this is just a rough starting point, that will evolve significantly! Some of the code is still pretty rough, however `src.model` and `src.dataloader` are useful places to start. 

## Example for training 

```
python train.py -dataPath ../train_data -runRootPath ../runs/test --preTrainedModelPath ../models/pytorch_vggish.pth -model AudioSet_fc_all -lr 0.0005
```

## Example for validation and testing 

See notebook `Evaluation.ipynb` (might be pretty rough, but should give a general idea)

## Example for processing unlabelled data with inference_and_chunk 

This is what is used when processing a round of unlabelled data, to generate candidates for annotation. 

```
python inference_and_chunk.py -wavMasterPath ..\data\wavmaster -sourceGuid WHOIS -modelPath ..\models\AudioSet_fc_all -positiveChunkDir ..\data\tmpPosChunks -positiveCandidatePredsDir ..\data\tmpPosPreds -positiveThreshold 0.2 -relativeBlobPath whoismasterchunked -negativeChunkDir ..\data\whoisnegativechunks -negativeThreshold 0.09
```
