# Setup instructions 

1. [Windows] Get [pyenv-win](https://github.com/pyenv-win/pyenv-win) to manage python versions:
    1. `git clone https://github.com/pyenv-win/pyenv-win.git %USERPROFILE%/.pyenv` 
    2. Add the following to your shell PATH `%USERPROFILE%\.pyenv\pyenv-win\bin`, `%USERPROFILE%\.pyenv\pyenv-win\shims` 

2. [Mac] Get [pyenv](https://github.com/pyenv/pyenv) to manage python versions:
	1. Use homebrew and run `brew update && brew install pyenv`
	2. Follow steps from #3 onwards [here](https://github.com/pyenv/pyenv#basic-github-checkout). Line #3 essentially adds the `pyenv init` command to your shell on startup 
	3. FYI this is a [commands reference](https://github.com/pyenv/pyenv/blob/master/COMMANDS.md)

3. [Common] Install and maintain the right Python version (3.6.8) 
    1. Run `pyenv --version` to check installation 
    2. Run `pyenv rehash` from your home directory, install python 3.6.8 with `pyenv install -l 3.6.8` (use 3.6.8-amd64 on Windows if relevant) and run `pyenv rehash` again 
    3. Cd to `/PodCast` and set a local python version `pyenv local 3.6.8` (or 3.6.8-amd64). This saves a `.python-version` file that tells pyenv-win what to use in this dir 
    4. Type `python --version` and check you're using the right one

4. Create a [virtual environment](https://docs.python.org/3.6/library/venv.html) to isolate and install package dependencies 
    1. In your working directory, run `python -m venv podcast-venv`. This creates a directory `podcast-venv` with relevant files/scripts. 
	2. On Mac, activate this environment with `source podcast-venv/bin/activate` and when you're done, `deactivate`
	   On Windows, activate with `.\podcast-venv\Scripts\activate.bat` and `.\podcast-venv\Scripts\deactivate.bat` when done
    3. In an active environment, cd to `/data_ml` and run `python -m pip install --upgrade pip && pip install -r requirements.txt` 

# Code for preparing WHOIS data and training/transfer learning from AudioSet 

Recommended directory structure:

```
PodCast
	- data_ml
	- models
		- pytorch_vggish.pth
	- runs 
	- train_data
		- wav
		- train.tsv
		- dev.tsv 
```

## Example for running train.py

```
python train.py -dataPath ../train_data -runRootPath ../runs/test --preTrainedModelPath ../models/pytorch_vggish.pth -model AudioSet_fc_all -lr 0.0005
```

## Example for processing unlabelled data with inference_and_chunk 

```
python inference_and_chunk.py -wavMasterPath ..\data\wavmaster -sourceGuid WHOIS -modelPath ..\models\AudioSet_fc_all -positiveChunkDir ..\data\tmpPosChunks -positiveCandidatePredsDir ..\data\tmpPosPreds -positiveThreshold 0.2 -relativeBlobPath whoismasterchunked -negativeChunkDir ..\data\whoisnegativechunks -negativeThreshold 0.09
```
