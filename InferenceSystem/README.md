# Run inference locally

Using Python 3 specifically tested with Python version 3.7.4

## Create a virtual environment

1. In your working directory, run `python -m venv inference-venv`. This creates a directory `inference-venv` with relevant files/scripts. 
2. On Mac, activate this environment with `source inference-venv/bin/activate` and when you're done, `deactivate`
    On Windows, activate with `.\inference-venv\Scripts\activate.bat` and `.\inference-venv\Scripts\deactivate.bat` when done
3. In an active environment, cd to `/InferenceSystem` and run `python -m pip install --upgrade pip && pip install -r requirements.txt` 

## Model download

1.  Download model from https://microsoft.sharepoint.com/:u:/t/OrcaCallAutomatedRecognitionSystemHackathon2019Project/EV9IBJrfmOxKhWtZOaW3pB8Br5u3yF0K3L18eZDruw89jw?e=S8UyEC
2.  Unzip model.zip and extract to `InferenceSystem/model`
3.  Check the contents of InferenceSystem/model
There should be 3 files
    Audioset_fc_all_*
    mean64.txt
    invstd64.txt

## Get connection string for interface with Azure Storage

1.  Go to the [Azure portal](https://portal.azure.com/)
Go to the `"LiveSRKWNotificationSystem"` resource group and within that go to the "livemlaudiospecstorage" storage account. Refer to [this page](https://docs.microsoft.com/en-us/azure/storage/blobs/storage-quickstart-blobs-python#copy-your-credentials-from-the-azure-portal) to see how to get the connection string.

### Windows

-------

```
setx AZURE_STORAGE_CONNECTION_STRING "<yourconnectionstring>"
```

### Mac or Linux

-------

```
export AZURE_STORAGE_CONNECTION_STRING="<copied-connection-string>"
```

## Get primary key for interface with CosmosDB

Go to the [Azure portal](https://portal.azure.com/)
Go to the `"LiveSRKWNotificationSystem"` resource group and within that go to the "aifororcasmetadatastore" cosmosdb account.

Go to "Keys" and look up the primary key

### Windows

-------

```
setx AZURE_COSMOSDB_PRIMARY_KEY "<yourprimarykey>"
```

### Mac or Linux

-------

```
export AZURE_COSMOSDB_PRIMARY_KEY="<yourprimarykey>"
```


## Run script

```
cd InferenceSystem
python LiveInferenceOrchestrator.py
```

You should see a bunch of .wav and .pngs written to the storage account.
You should see entries in the CosmosDB.

# Running inference system in a local Docker container

## Prerequisites

- **Docker**: To complete this step, you need Docker installed locally.  Docker provides packages that configure the Docker
environment on 
[macOS](https://docs.docker.com/docker-for-mac/),
[Windows](https://docs.docker.com/docker-for-windows/), and
[Linux](https://docs.docker.com/engine/installation/#supported-platforms).

- **model.zip**: Download model from 
[this link](https://microsoft.sharepoint.com/:u:/t/OrcaCallAutomatedRecognitionSystemHackathon2019Project/EV9IBJrfmOxKhWtZOaW3pB8Br5u3yF0K3L18eZDruw89jw?e=S8UyEC).
Save the model to `InferenceSystem/model.zip`.

- **Environment Variable File**: Create/get an environment variable file.  This should be a file called `inference-system/.env`.
This can be completed in two ways.
1.  Ask an existing contributor for their .env file.
2.  Create one of your own.  This .env file should be created in the format below.
`<key>` and `<string>` should be filled in with the Azure Storage Connection String and the Azure CosmosDB Primary Key above.

```
AZURE_COSMOSDB_PRIMARY_KEY=<key>
AZURE_STORAGE_CONNECTION_STRING=<string>
```

## Building the docker container

From the `InferenceSystem` directory, run the following command.
It will take a while (~2-3 minutes on macOS or Linux, ~10-20 minutes on Windows) the first time, but builds are cached, and it
should take a much shorter time in future builds.

```
docker build . -t live-inference-system -f ./FastAIDocker/Dockerfile
```

## Running the docker container

From the `InferenceSystem` directory, run the following command.

```
docker run --rm -it --env-file .env live-inference-system
```

You should see some .wav and .png files written to the storage account, and entries in CosmosDB.
In addition, you should see something similar to the following in your console.

```
Loaded checkpoint: model/AudioSet_fc_all_Iter_22
Listening to location https://s3-us-west-2.amazonaws.com/streaming-orcasound-net/rpi_orcasound_lab
Downloading live982.ts
live982.ts: 205kB [00:00, 653kB/s]
Downloading live983.ts
live983.ts: 205kB [00:00, 589kB/s]
Downloading live984.ts
live984.ts: 205kB [00:00, 664kB/s]
Downloading live985.ts
live985.ts: 205kB [00:00, 604kB/s]
Downloading live986.ts
live986.ts: 205kB [00:00, 641kB/s]
Downloading live987.ts
live987.ts: 205kB [00:00, 640kB/s]
Loading file: 2020-07-27T16:14:54.322546.wav
Uploaded audio to Azure
Uploaded spectrogram to Azure
Added metadata to cosmos db

```

# Deploying an updated docker build to Azure Container Instances

## Prerequisites

- You must have completed all of the steps above: Running inference system in a local Docker container.  You should have a 
container that is working locally that you wish to deploy live to production.

- **Azure CLI**: You must have Azure CLI version 2.0.29 or later installed on your local computer. Run az --version to find the 
version. If you need to install or upgrade, see 
[Install the Azure CLI](https://docs.microsoft.com/en-us/cli/azure/install-azure-cli).

## Pushing your image to Azure Container Registry

This step pushes your local container to the Azure Container Registry (ACR).  If you would like more information, this
documentation is adapted from 
[this tutorial](https://docs.microsoft.com/en-us/azure/container-instances/container-instances-tutorial-prepare-acr).

Login to the shared azure directory from the Azure CLI.

```
az login --tenant ai4orcasoutlook.onmicrosoft.com
```

We will be using the orcaconservancycr ACR in the LiveSRKWNotificationSystem Resource Group.

Log in to the container registry.

```
az acr login --name orcaconservancycr
```

You should receive something similar to `Login succeeded`.

Tag your docker container with the version number.  For information on what version number to use, see 
[this article](https://semver.org/).

```
docker tag live-inference-system orcaconservancycr.azurecr.io/live-inference-system:v<Major>.<Minor>.<Patch>
```

Lastly, push your image to Azure Container Registry.

```
docker push orcaconservancycr.azurecr.io/live-inference-system:v<Major>.<Minor>.<Patch>
```

## Deploying your updated container to Azure Container Instances

Edit the file `InferenceSystem/deploy-aci.yaml`.  There are three sensitive strings that must be filled in before deployment can
happen.

**NOTE** - Make sure you change these back after running the build - don't commit them to the repository!

1.  `<cosmos_primary_key>` - Replace this with the AZURE_COSMOSDB_PRIMARY_KEY from your .env file (or found above).
2.  `<storage_connection_string>` - Replace this with the AZURE_STORAGE_CONNECTION_STRING from your .env file (or found above).
3.  `<image_registry_password>` - Replace this with the password for the orcaconservancycr container registry.  It can be found at
[this link](https://portal.azure.com/#@OrcaConservancy778.onmicrosoft.com/resource/subscriptions/9ffa543e-3596-43aa-b82c-8f41dfbf03cc/resourcegroups/LiveSRKWNotificationSystem/providers/Microsoft.ContainerRegistry/registries/orcaconservancycr/accessKey)
under the name `password`.

Then, run this command from the `InferenceSystem` directory.  It will take a while to complete.  Once complete, make sure to check your work below.

```
az container create -g LiveSRKWNotificationSystem -f .\deploy-aci.yaml
```

## Checking your work

View the container logs with the following command.  The logs should be similar to the logs created when you run the container locally (above).

```
az container attach --resource-group LiveSRKWNotificationSystem --name live-inference-system-aci-3gb
```

# No changes made to deploy-aci.yaml?

I purposefully told git to ignore all futher changes to the file with this command: `git update-index --assume-unchanged deploy-aci.yaml`.  This is to prevent people from thecking in their credentials into the repository.  If you want t a change to be tracked, you can turn off this feature with `git update-index --no-assume-unchanged deploy-aci.yaml`


# Automatic annotation data upload script PrepareDataForPredictionExplorer.py

Find the connection string for blob storage account mldevdatastorage and run the following 

### Windows

-------

```
setx PODCAST_AZURE_STORAGE_CONNECTION_STRING "<yourconnectionstring>"
```

### Mac or Linux

-------

```
export PODCAST_AZURE_STORAGE_CONNECTION_STRING="<copied-connection-string>"
```

Call the script

```
python PrepareDataForPredictionExplorer.py --start_time "2020-07-25 19:15" --end_time "2020-07-25 20:15" --s3_stream https://s3-us-west-2.amazonaws.com/streaming-orcasound-net/rpi_orcasound_lab --model_path <folder> --annotation_threshold 0.4 --round_id round5 --dataset_folder <path-to-folder>
```
