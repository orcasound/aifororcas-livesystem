import argparse
import pandas as pd
import numpy as np
import librosa
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from tqdm import tqdm
from sklearn.metrics import recall_score, precision_score, \
    f1_score, average_precision_score, precision_recall_curve


def quantize_interval_to_seconds(startTime, duration, maxDuration):
    """
    Returns a list of integers containing corresponding seconds. 
    If second N:N+1 contains part of the interval, N is counted. 
    """
    endTime = startTime + duration
    low = int(np.floor(startTime))
    high = min(int(np.ceil(endTime)), int(maxDuration))
    seconds = list(range(low, high+1))
    return seconds

def quantize_interval_df(df, startColumn, durationColumn, confColumn, maxDuration, threshold=None):
    '''
    Convert given intervals into a 1 second quantized examples for scoring 
    '''
    df = df.sort_values(startColumn).reset_index(drop=True)
    timeWindows = []
    for idx in range(df.shape[0]):
        startTime = df.loc[idx,startColumn]
        duration = df.loc[idx,durationColumn]
        confidence = 1.0 if confColumn is None else df.loc[idx, confColumn]
        
        for time_idx in quantize_interval_to_seconds(startTime, duration, maxDuration):
            timeWindows.append((time_idx, confidence))

    # unique operation merges overlapping windows 
    timeWindows = sorted(list(set(timeWindows)))
    idxs, confidences = zip(*timeWindows)

    if threshold is None:
        positiveIdxs = idxs
    else:
        positiveIdxs = [tup[0] for tup in timeWindows if tup[1] > threshold]
    
    ## Create dataframe quantized into 1-second time windows for scoring 
    quantized_df = pd.DataFrame({
        'timewindow': range(int(np.ceil(maxDuration))),
        'label': 0, 
        'confidence': 0.0
        })
    quantized_df.loc[positiveIdxs,'label'] = 1
    quantized_df.loc[idxs,'confidence'] = confidences

    return quantized_df

def score_quantized_examples(dataset, submissionQuantized, groundTruthQuantized, threshold):

    ## Evaluating
    precision, recall, thresholds = precision_recall_curve(groundTruthQuantized.label, submissionQuantized.confidence)
    class_prevalence = precision[0]

    metrics = dict()
    if threshold is not None:
        metrics['recall'] = round(recall_score(groundTruthQuantized.label, submissionQuantized.label), 3)
        metrics['precision'] = round(precision_score(groundTruthQuantized.label, submissionQuantized.label), 3)
        metrics['f1_score'] = round(f1_score(groundTruthQuantized.label, submissionQuantized.label), 3)

    metrics['AUPRC'] = round(average_precision_score(groundTruthQuantized.label, submissionQuantized.confidence), 3)

    auprc_curve = pd.DataFrame(dict(
        precision=precision,
        recall=recall,
        thresholds=[0.0, *thresholds]
    ))
    
    return metrics, auprc_curve, class_prevalence

def score_submission(testSetDir, submissionFile, threshold=None, verbose=False):
    # load Ground truth and submission 
    testData = pd.read_csv(Path(testSetDir)/"test.tsv", delimiter='\t')
    testWavDir = Path(testSetDir)/"wav"
    submissionName = Path(submissionFile).stem
    submissionData = pd.read_csv(submissionFile, sep='\t')

    # iterate over (test set, wav file) -> aggregate scores appropriately
    # append for all wav files in a test set and score
    metrics_list = []
    auprc_curve_list = []
    for group in testData.groupby('dataset'):
        dataset, datasetGroundTruth = group

        print("\n###\nScoring dataset:", dataset)
        gt_list, sub_list = [], []
        total_duration = 0.0
        for wavGroup in tqdm(datasetGroundTruth.groupby('wav_filename')):
            wav_filename, groundTruth = wavGroup

            # retrieve intervals for this group (labels, submission)
            # TODO@Akash: remove this dependency to require the audio
            max_length = librosa.get_duration(filename=str(testWavDir/wav_filename))
            # Convert Ground Truth: HACK doesn't have confidence column, so passing dummy value  
            gt_list.append(quantize_interval_df(
                    groundTruth, 'start_time_s', 'duration_s', 'duration_s', max_length
                ))

            # Quantize submission file into 1-second time windows (timewindow, label, confidence)
            submission = submissionData.query('wav_filename == @wav_filename')
            sub_list.append(quantize_interval_df(
                submission, 'start_time_s', 'duration_s', 'confidence', max_length, threshold=threshold
                ))
            
            total_duration += max_length
        
        groundTruthQuantized = pd.concat(gt_list)
        submissionQuantized = pd.concat(sub_list)
        print("Total duration: {:.0f}:{:.0f}".format(total_duration//60, total_duration%60))
        if verbose:
            print("\nSnippet of converted/quantized ground truth file\n", groundTruthQuantized.head(3))
            print("\nSnippet of converted/quantized submission file\n", submissionQuantized.head(3))

        # score and aggregate results 
        metrics, auprc_curve, class_prevalence = score_quantized_examples(
            dataset, submissionQuantized, groundTruthQuantized, threshold
            )
        metrics_list.append({'dataset':dataset, **metrics})
        auprc_curve['dataset'] = dataset
        auprc_curve_list.append(auprc_curve)
    
    metrics = pd.DataFrame.from_records(metrics_list)
    overall = metrics.mean(numeric_only=True)
    overall['dataset'] = 'OVERALL'
    metrics = metrics.append(overall, ignore_index=True).round(3)
    
    return metrics, pd.concat(auprc_curve_list)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-testSetDir', default=None, type=str, required=True)
    parser.add_argument('-submissionFiles', default=None, type=str, required=True)
    parser.add_argument('-threshold', default=None, type=float, required=False)
    args = parser.parse_args()

    submissionFiles = args.submissionFiles.split(',')
    metricsFile = Path(submissionFiles[0]).parent/"metrics.tsv"
    resultsFile = Path(submissionFiles[0]).parent/"results.md"
    plotsFile = Path(submissionFiles[0]).parent/"{}.png".format("au_pr_curves")

    # scoring multiple submission files 
    metrics_list, auprc_list = [], []
    for submissionFile in submissionFiles:
        metrics, auprc_curve = score_submission(args.testSetDir, submissionFile, args.threshold)
        # TODO@Akash: include class prevalence as a "no-skill" submission 
        metrics['submission'] = Path(submissionFile).stem
        metrics_list.append(metrics)
        auprc_curve['submission'] = Path(submissionFile).stem
        auprc_list.append(auprc_curve)
    
    metrics = pd.concat(metrics_list)
    metrics.to_csv(metricsFile, sep='\t', index=False)
    print("Metrics written to", metricsFile)
    print(metrics.set_index(['submission', 'dataset']))
    metrics_table = metrics.pivot(index='dataset', columns='submission', values='AUPRC')
    with open(resultsFile, 'w') as f:
        f.write(metrics_table.to_markdown())
        print("Results summary written to", resultsFile)
    
    # aggregate and compile results from different submission files 
    auprc_curve = pd.concat(auprc_list)
    p = sns.FacetGrid(data=auprc_curve, col='dataset', row='submission', margin_titles=True)
    p.map(sns.lineplot, 'recall', 'precision')
    p.set(ylim=(0.,1.0))
    p.set_titles(col_template="{col_name}", row_template="{row_name}")
    plt.savefig(plotsFile)
    print("Precision-Recall plots written to", plotsFile)
