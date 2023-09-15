# Humpback Whales Vocalizations Classification

## Overview

This folder contains the collaborative efforts of the team for the Microsoft Global Hackathon 2023. The project focuses on the [classification of humpback whale vocalizations](https://www.orcasound.net/portfolio/humpback-catalogue/) using advanced deep learning techniques. Within this folder, we collected a series of Jupyter notebook organized into sub-folders based on their specific role in solving the classification problem.

### References:
- **Data Source**: The raw audio data is sourced from the ["Haro Humpback" catalog & open annotations](s3://acoustic-sandbox/humpbacks/Emily-Vierling-Orcasound-data/Em_HW_Processed/) prepared by Emily Vierling.
- **Denoising Toolkit**: The project utilizes [ORCA-CLEAN: A Deep Denoising Toolkit for Killer Whale Communication](https://www.isca-speech.org/archive/interspeech_2020/bergler20_interspeech.html), developed by Christian Bergler. The toolkit can be found [here](https://github.com/ChristianBergler/ORCA-CLEAN).
- **Audio Spectrogram Transformer**: The project also appies the [Audio Spectrogram Transformer](https://github.com/YuanGongND/ast) developed by Yuan Gong, Yu-An Chung, and James Glass.


## Table of Contents

- [Data](#data)
- [Installation](#installation)
- [Running Notebooks on Azure ML](#running-notebooks-on-azure-ml)
- [Folder Organization](#folder-organization)
- [Checklist](#checklist)

## Prerequisites
- An Azure account with an active subscription - [Create an account for free](https://azure.microsoft.com/free/?WT.mc_id=A261C142F)


## Data

The dataset consists of audio files of humpback whale vocalizations and corresponding labels. The raw data is taken from the ["Haro Humpback" catalog and open annotations prepared by Emily Vierling](https://github.com/orcasound/orcadata/wiki/Other-training-data:-humpback-whales). For more details about the data used in this project, refer to the [relevant Orcasound GitHub repository](https://github.com/orcasound/orcadata/wiki/Other-training-data:-humpback-whales).


## Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/LianaN/humpback-whales-vocalizations-classification.git
    ```
2. Navigate to the `HumpbackWhales` folder:
    ```bash
    cd humpback-whales-vocalizations-classification/HumpbackWhales
    ```


## Running Notebooks on Azure ML

1. **Azure ML Workspace Setup**: If an Azure ML workspace is not already available, it can be created by following the instructions [here](https://learn.microsoft.com/en-us/training/modules/create-workspace-resources-getting-started-azure-machine-learning/5-create-azure-machine-learning-workspace).
2. **Notebook Upload**: From Azure portal navigate to the Overview tab of Azure Machine Learning workspace, and click on `Launch studio`. In Azure AI Studio go to the "Notebooks" tab, and upload the Jupyter notebooks from the respective sub-folders.
3. **Compute Target Creation**: For notebook execution, a Compute target must be established. This can be done via the "Compute" tab within the Azure ML workspace. A compute size of `STANDARD_D13_V2` or equivalent is generally sufficient for running most notebooks.
4. **Dependency Installation**: Open a notebook and execute the cell containing the required package installations. These are specified at the beginning of each notebook.
5. **Notebook Execution**: Read the instructions in the Header of a notebook for additional setup, if any. Run the notebook cells sequentially to execute the code.


## Folder Organization

- `01_DataPreprocessing`: Notebooks for data acquisition and preprocessing.
- `02_ExploratoryDataAnalysis`: Notebooks for exploratory data analysis.
- `03_Denoising/ML_pipeline`: Notebooks and scripts for setting up Azure ML pipelines for training the [ORCA-CLEAN deep denoising network](https://github.com/ChristianBergler/ORCA-CLEAN).
- `04_Classification`: Notebooks for running classification experiments.
- `05_ModelEvaluation`: Notebooks for evaluating the performance of the models.

## Checklist

### Done
- [x] Acquisition and annotation-based extraction of hydrophone recordings: download raw hydrophone recordings from Orcasound Amazon S3 and extract humpback vocalizations based on provided annotations.
- [x] Noise segment isolation: develop notebooks to isolate noise segments from raw hydrophone recordings for further analysis.
- [x] AzureML pipeline integration for ORCA-CLEAN: encapsulate the training process of the ORCA-CLEAN denoising model within an AzureML pipeline.
- [x] Performance benchmarking of Audio Spectrogram Transformer: conduct basic benchmarking tests for the Audio Spectrogram Transformer on humpback vocalizations.

### ToDo
- [ ] Noise segment refinement: develop a binary classifier to accurately distinguish between true noise segments and those erroneously labeled as noise but containing vocalizations.
- [ ] Additional testing data preparation: develop a notebook for extracting %-portions from vocalizations using a sliding window of 5 sconds.
- [ ] ORCA-CLEAN model optimization: finalize the training of the ORCA-CLEAN model through iterative training and testing cycles for humpback vocalizations.
- [ ] Data denoising: utilize the trained ORCA-CLEAN model to preprocess the vocalization data, generating denoised records for subsequent steps.
- [ ] Classifier fine-tuning: fine-tune the Audio Spectrogram Transformer using the denoised records to improve classification performance.
- [ ] Inference pipeline deployment: develop and deploy an end-to-end inference pipeline comprising two sequential steps: denoising using ORCA-CLEAN and classification using the Audio Spectrogram Transformer.