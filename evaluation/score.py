import argparse
import pandas as pd
import numpy as np
import librosa
from sklearn.metrics import recall_score, precision_score, f1_score




## Validation Data generator
def dataGenerator(df, startColumn, durationColumn, maxDuration):
    '''
    Creating 1 second interval-label file for evaluation
    '''
    positiveLabel = []
    for idx in range(df.shape[0]):
        ## Lowest start time
        startTime = df.loc[idx,startColumn]
        start = int(np.floor(startTime))
        duration = df.loc[idx,durationColumn]
        
        ## Highest end time
        end = int(np.ceil(startTime + duration) + 1)
        for idx in range(start,end):
            positiveLabel.append(idx)
    ## Unique
    positiveLabel = list(set(positiveLabel))
    
    ## Create final data
    validation_df = pd.DataFrame({'timewindow':range(int(np.ceil(maxDuration))),'label':0})
    validation_df.loc[positiveLabel,'label']=1
    return validation_df

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-audioFilePath', default=None, type=str, required=True)
    parser.add_argument('-groundTruthFilePath', default=None, type=str, required=True)
    parser.add_argument('-submissionFilePath', default=None, type=str, required=True)
    args = parser.parse_args()

    wavFilePath = args.audioFilePath
    max_length = librosa.get_duration(filename=wavFilePath)
    

    ##load Ground truth
    testData = pd.read_csv(args.groundTruthFilePath, delimiter='\t')

    ## Convert Ground Truth
    groundTruth = dataGenerator(testData, 'start_time_s', 'duration_s', max_length)

    ## Load Sample submission
    sampleSub = pd.read_csv(args.submissionFilePath)

    ## Convert sample file
    output_df = dataGenerator(sampleSub, 'StartTime', 'Duration', max_length)


    ## Evaluating
    print('Recall:{0} \nPrecision:{1} \nF-1 Score:{2}'.format(
        recall_score(groundTruth.label, output_df.label),
        precision_score(groundTruth.label, output_df.label),
        f1_score(groundTruth.label, output_df.label)))