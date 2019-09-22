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
