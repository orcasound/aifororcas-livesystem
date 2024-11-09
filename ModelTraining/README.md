# Repo to explain FastAI modeling methodology
- Author : Aayush Agrawal (aaagraw@microsoft.com)

## Setup

Author: Bruno Grande

The following instructions describe how I was able to install the dependencies for these Jupyter notebooks. I was running into a version conflict with `pip install`. I was able to resolve dependencies using `uv pip install`.

```console
# Navigate to ModelTraining subdirectory
cd aifororcas-livesystem/ModelTraining

# Create a new conda environment with Python 3.8
conda create -n <env-name> python=3.8
conda activate <env-name>

# Install uv (better at resolving package version conflicts)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install dependencies using `uv pip` (instead of plain `pip`)
uv pip install -r requirements.txt
```

I'm also including a full list of installed packages and versions in `requirements.lock.txt`, which was generated using `pip freeze`.

## Model data 
The base data used here is hosted on Current test set for evaluation is hosted on [Orca Sound website ](https://github.com/orcasound/orcadata/wiki/Pod.Cast-data-archive#test-sets). The model was trained with the following dataset -

- [WHOIS09222019_PodCastRound1](https://github.com/orcasound/orcadata/wiki/Pod.Cast-data-archive#WHOIS_PodCastRound1) (~6hrs, open data source – Orca call around the planet, Good for generic models)
- [OrcasoundLab07052019_PodCastRound2](https://github.com/orcasound/orcadata/wiki/Pod.Cast-data-archive#OrcasoundLab07052019_PodCastRound2) (~1.2hrs, live hydrophone data - SRKW)
- [OrcasoundLab09272017_PodCastRound3](https://github.com/orcasound/orcadata/wiki/Pod.Cast-data-archive#OrcasoundLab09272017_PodCastRound3) (~1.6hrs, live hydrophone data - SRKW)

And testing in evaluation section of FastAI start script is done on this data - 
- [OrcasoundLab07052019_Test](https://github.com/orcasound/orcadata/wiki/Pod.Cast-data-archive#OrcasoundLab07052019_Test) (~30min, very call-rich, good SNR conditions)
- [OrcasoundLab09272017_Test](https://github.com/orcasound/orcadata/wiki/Pod.Cast-data-archive#OrcasoundLab07052019_Test) (~21min, more typical distribution, good SNR conditions)

## Data Directory structure in this folder(not uploaded on Git) -

```
fastai/data
    - train
        - train_data_09222019
            - wav
            - train.tsv
        - Round2_OS_07_05
            - wav
            - train.tsv
        - Round3_OS_09_27_2017
            - wav
            - train.tsv
        - mldata
            - all
                - models
                - negative
                - positive

    - test
        - all
        - OrcasoundLab07052019_Test
            - test2Sec
            - wav
            - test.tsv
        - OrcasoundLab09272017_Test
            - wav
            - test.csv
```

## End-To-End development Process

### **Step 1 - OPTIONAL**: Creating a more ML-ready dataset inline with other popular sound dataset - [1_DataCreation.ipynb notebook]()
- NOTE: For Quick Start, you may see *.wav files in train/mldata/all/[negative|positive] and in test/all/[negative|positive]; in this case, there's no need to run this script (i.e. no need to generate new samples)
- Extracting small audio segments for positive and negative label and store them in positive and negative folder for training  (Filtering any call with less than 1 second duration)
- Also create 2 sec audio sample from testing data for formal ML evaluation

### **Step2**: Transfer leaning a ResNet18 model with on the fly spectogram creation with frequency masking for data augmentation - [2_FastAI_StarterScript.ipynb]()
- Step 1: Create a Dataloader using FastAI library (Defining parameters to create a spectogram)
- Step 2: Create a DataBunch by splitting training data in 80:20% for validation using training phase, also defining augmentation technique (Frequency transformation)
- Step 3: Training a model using ResNet18 (pre-trained on ImageNet Data)
- Step 4: Running evaluation on Test set
- Step 5: Running scoring for official evaluation

### **Step3**: Inference Testing[3_InferenceTesting.ipynb]()
Notebook showing how to use the inference.py to load the model and do predictions by defining a clip path.
The inference.py returns a dictionary -
- **local_predictions**: A list of predictions (1/0) for each second of wav file (list of integers)
- **local_confidences**: A list of probability of the local prediction being a whale call or not (list of float)
- **global_predictions**: A global prediction(1/0) for the clip containing a valid whale call (integer)
- **global_confidences**: Probability of the prediction being a whale call or not(float)

# All the files generated or used during training are [here:](https://portal.azure.com/#@adminorcasound.onmicrosoft.com/resource/subscriptions/c65c3881-6d6b-4210-94db-5301ef484f17/resourceGroups/mldev/providers/Microsoft.Storage/storageAccounts/storagesnap/overview). NOTE: You must hold "Contributor" role assignment in the OrcaSound Azure tenant to access.
- **Models/stg2-rn18.pkl** - Trained ResNet18 model
- **mldata.7z** is data used for training the model
- **All.zip** is testing data for early evaluations
- **test2Sec.zip** is data used for scoring for official evaluations

## Dependencies -
- [FastAIAudio](https://github.com/mogwai/fastai_audio)
- [FastAI v1](https://github.com/fastai/fastai/blob/master/README.md#installation)
- [Librosa] (https://github.com/librosa/librosa)
- [Pandas] (https://github.com/pandas-dev/pandas)
- [Pydub] (https://github.com/jiaaro/pydub/)
- [tqdm] (https://github.com/tqdm/tqdm)
- [sklearn](https://scikit-learn.org/stable/install.html)
