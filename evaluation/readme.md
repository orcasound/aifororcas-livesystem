# Model Evaluation

Current test set for evaluation is hosted on [Orca Sound website ](https://github.com/orcasound/orcadata/wiki/Pod.Cast-data-archive#test-sets). 

## Sample Submission format 
We need a csv with two columns -
- **StartTime** - The start time in the test audio file(in sec) where the model detected an orca call
- **Duration** - The duration from the start time till which you detected a call

**Example of a submission file -**

| StartTime       | Duration    | 
| :------------- | :----------: | 
|  22.17 | 1.12   | 
| 24.87   | 1.10 | 

We also have a sample submission file for your perusal.

## Methodology -
We divide the audio in one second chunks and label it 1/0 based on the ground truth labeling as an Orca call detected or not respectively. After this we use [F-1 score](https://en.wikipedia.org/wiki/F1_score) to evaluate model performance.


## How to run the script

Create a file structure like below -

Evaluation
- groundTruth
    - test.tsv
- wav
    - OS_7_05_2019_08_24_00_.wav
- sample_submission.csv
- score.py


```bash
python score.py -audioFilePath "./wav/OS_7_05_2019_08_24_00_.wav" -groundTruthFilePath "./groundTruth/test.tsv" -submissionFilePath "./sample_submission.csv"
```