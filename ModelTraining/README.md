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

I'm also including a full list of installed packages and versions below, which was generated using `pip freeze`.

<details>
    <summary>Click to expand the full list of installed packages and versions</summary>

```plaintext
annotated-types==0.7.0
asttokens==2.4.1
attrs==24.2.0
audioread==3.0.1
backcall==0.2.0
beautifulsoup4==4.12.3
bleach==6.1.0
blis==0.7.11
Bottleneck==1.4.0
catalogue==2.0.10
certifi==2024.8.30
cffi==1.17.1
charset-normalizer==3.4.0
click==8.1.7
cloudpathlib==0.20.0
comm==0.2.2
confection==0.1.5
contourpy==1.1.1
cycler==0.12.1
cymem==2.0.8
debugpy==1.8.8
decorator==5.1.1
defusedxml==0.7.1
docopt==0.6.2
executing==2.1.0
fastai==1.0.61
fastai_audio @ git+https://github.com/fastaudio/fastai_audio@3730194a0ed14e142416f60e71effa6e28058b60
fastjsonschema==2.20.0
fastprogress==1.0.3
fire==0.7.0
fonttools==4.54.1
future==1.0.0
idna==3.10
importlib_metadata==8.5.0
importlib_resources==6.4.5
ipykernel==6.29.5
ipython==8.12.3
jedi==0.19.1
Jinja2==3.1.4
joblib==1.4.2
jsonschema==4.23.0
jsonschema-specifications==2023.12.1
jupyter_client==8.6.3
jupyter_core==5.7.2
jupyterlab_pygments==0.3.0
kiwisolver==1.4.7
langcodes==3.4.1
language_data==1.2.0
lazy_loader==0.4
librosa==0.10.0
llvmlite==0.41.1
marisa-trie==1.2.1
markdown-it-py==3.0.0
MarkupSafe==2.1.5
matplotlib==3.7.5
matplotlib-inline==0.1.7
mdurl==0.1.2
mistune==3.0.2
msgpack==1.1.0
murmurhash==1.0.10
nbclient==0.10.0
nbconvert==7.16.4
nbformat==5.10.4
nest-asyncio==1.6.0
numba==0.58.1
numexpr==2.8.6
numpy==1.24.4
nvidia-ml-py3==7.352.0
packaging==24.2
pandas==2.0.3
pandocfilters==1.5.1
parso==0.8.4
pexpect==4.9.0
pickleshare==0.7.5
pillow==10.4.0
pipreqs==0.5.0
pkgutil_resolve_name==1.3.10
platformdirs==4.3.6
pooch==1.8.2
preshed==3.0.9
prompt_toolkit==3.0.48
psutil==6.1.0
ptyprocess==0.7.0
pure_eval==0.2.3
pycparser==2.22
pydantic==2.9.2
pydantic_core==2.23.4
pydub==0.24.1
Pygments==2.18.0
pyparsing==3.1.4
python-dateutil==2.9.0.post0
pytz==2024.2
PyYAML==6.0.2
pyzmq==26.2.0
referencing==0.35.1
requests==2.32.3
rich==13.9.4
rpds-py==0.20.1
scikit-learn==1.3.2
scipy==1.10.1
shellingham==1.5.4
six==1.16.0
smart-open==7.0.5
soundfile==0.12.1
soupsieve==2.6
soxr==0.3.7
spacy==3.7.5
spacy-legacy==3.0.12
spacy-loggers==1.0.5
srsly==2.4.8
stack-data==0.6.3
termcolor==2.4.0
thinc==8.2.5
threadpoolctl==3.5.0
tinycss2==1.4.0
torch==1.6.0+cu92
torchaudio==0.6.0
torchvision==0.7.0+cu92
tornado==6.4.1
tqdm==4.67.0
traitlets==5.14.3
typer==0.13.0
typing_extensions==4.12.2
tzdata==2024.2
urllib3==2.2.3
wasabi==1.1.3
wcwidth==0.2.13
weasel==0.4.1
webencodings==0.5.1
wrapt==1.16.0
yarg==0.1.9
zipp==3.20.2
```

</details>

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
