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

## Example for inference_and_chunk

```
python inference_and_chunk.py -wavMasterPath ..\data\OS_7_05_2019_Curated -modelPath ..\models\AudioSet_fc_all -outputChunkDir ..\data\OS_test_chunks -outputPredsDir ..\data\OS_test_chunks_preds
```

