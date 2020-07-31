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

## Using Existing Model
We used existing model saved [here](https://microsoft-my.sharepoint.com/:f:/p/aaagraw/ErYhp5U7TvNLuEm_D1sGCfkBHxx_h9ACA4Yn7WbfK5dzlA?e=V5x9b5) built by Aakash in data_ml folder. We used ScoringUsingExistingModel.ipynb file to generate submision/predictionSubmission.csv and used score.py to evalute against test data.

| File       | Model       | Recall    | Precision    | F-1 Score| Threshold |
| :------------- | :----------: |  :----------: |  :----------: | :----------: | :----------: | 
|  sample_submission.csv | None | 0.7168674698795181 | 0.7168674698795181 | 0.7168674698795182 | N/A |
| submission/predictionSubmission.csv | [AudioSet_fc_all_Iter_26](https://microsoft-my.sharepoint.com/:f:/p/aaagraw/ErYhp5U7TvNLuEm_D1sGCfkBHxx_h9ACA4Yn7WbfK5dzlA?e=3bbeI9) | 0.4569707401032702 | 0.9448398576512456 | 0.6160092807424594| 0.5 |
|  submission/submission2SecFastAI.csv | [stg2-rn18.pkl](https://microsoft-my.sharepoint.com/:u:/p/aaagraw/EQBViKhKgUxPgEJMHZlVY2wBn_zyipowNxUL_VSzzEidRA?e=zfSqVe) | 0.9130808950086059 | 0.0.9381078691423519 | 0.9254252071522024 | 0.5|
