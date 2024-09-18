# Working with the InferenceSystem

The InferenceSystem is an umbrella term for all the code used to stream audio from Orcasound's S3 buckets, perform inference on audio segments using the deep learning model and upload positive detections to Azure. The entrypoint for the InferenceSystem is [src/LiveInferenceOrchestrator.py](src/LiveInferenceOrchestrator.py).

This document describes the following steps
1. How to run the InferenceSystem locally.
2. Deploying an updated docker build to Azure Container Instances.

Note: We use Python 3, specifically tested with Python 3.7.4

# How to run the InferenceSystem locally
## Create a virtual environment

1. In your working directory, run `python -m venv inference-venv`. This creates a directory `inference-venv` with relevant files/scripts. 
2. On Mac or Linux, activate this environment with `source inference-venv/bin/activate` and when you're done, `deactivate`

    On Windows, activate with `.\inference-venv\Scripts\activate.bat` and `.\inference-venv\Scripts\deactivate.bat` when done
3. In an active environment, cd to `/InferenceSystem` and run `python -m pip install --upgrade pip && pip install -r requirements.txt` 

## Model download

1.  Download the current production model from [this link.](https://trainedproductionmodels.blob.core.windows.net/dnnmodel/11-15-20.FastAI.R1-12.zip)
2.  Unzip *.zip and extract to `InferenceSystem/model` using `unzip 11-15-20.FastAI.R1-12.zip`
3.  Check the contents of `InferenceSystem/model`.
There should be 1 file
    * model.pkl

## Get connection string for interface with Azure Storage
To be able to upload detections to Azure, you will need a connection string.

Go to [Azure portal](https://portal.azure.com/) and find the `"LiveSRKWNotificationSystem"` resource group. Within that go to the `"livemlaudiospecstorage"` storage account. Refer to [this page](https://docs.microsoft.com/en-us/azure/storage/blobs/storage-quickstart-blobs-python#copy-your-credentials-from-the-azure-portal) to see how to get the connection string.

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

Go to the `"LiveSRKWNotificationSystem"` resource group and within that go to the `"aifororcasmetadatastore"` CosmosDB account.

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

## Get connection string for interface with App Insights

Go to the [Azure portal](https://portal.azure.com/)

Go to the `"LiveSRKWNotificationSystem"` resource group and within that go to the `"InferenceSystemInsights"` App Insights service

Look up the connection key from 'Essentials'

### Windows

-------

```
setx INFERENCESYSTEM_APPINSIGHTS_CONNECTION_STRING "<yourconnectionstring>"
```

### Mac or Linux

-------

```
export INFERENCESYSTEM_APPINSIGHTS_CONNECTION_STRING="<yourconnectionstring>"
```

## Run live inference locally

```
cd InferenceSystem
python src/LiveInferenceOrchestrator.py --config ./config/Test/FastAI_LiveHLS_OrcasoundLab.yml
```

You should see the following logs in your terminal. Since this is a Test config, no audio is uploaded to Azure and no metadata is written to CosmosDB.

```
Listening to location https://s3-us-west-2.amazonaws.com/streaming-orcasound-net/rpi_orcasound_lab
Downloading live879.ts
live879.ts: 205kB [00:00, 1.17MB/s]                                             
Downloading live880.ts
live880.ts: 205kB [00:00, 1.11MB/s]                                             
Downloading live881.ts
live881.ts: 205kB [00:00, 948kB/s]                                              
Downloading live882.ts
live882.ts: 205kB [00:00, 1.14MB/s]                                             
Downloading live883.ts
live883.ts: 205kB [00:00, 1.07MB/s]                                             
Downloading live884.ts
live884.ts: 205kB [00:00, 1.04MB/s]                                             
rpi_orcasound_lab_2021_10_13_15_11_18_PDT.wav
Length of Audio Clip:60.010666666666665
Preprocessing: Downmixing to Mono
Preprocessing: Resampling to 200009/59 00:00<00:00]
```

# Running inference system in a local docker container

To deploy to production we use Azure Container Instances. To enable deploying to production, you need to first build the docker image for the inference system locally.

## Prerequisites

- **Docker**: To complete this step, you need Docker installed locally.  Docker provides packages that configure the Docker
environment on 
[macOS](https://docs.docker.com/docker-for-mac/),
[Windows](https://docs.docker.com/docker-for-windows/), and
[Linux](https://docs.docker.com/engine/installation/#supported-platforms).

- **model.zip**: Download model from 
[this link](https://trainedproductionmodels.blob.core.windows.net/dnnmodel/11-15-20.FastAI.R1-12.zip).
Rename the `*.zip` to `model.zip` and place it in `InferenceSystem/model.zip`.

- **Environment Variable File**: Create/get an environment variable file.  This should be a file called `inference-system/.env`.
This can be completed in two ways.
    1.  Ask an existing contributor for their .env file.
    2.  Create one of your own.  This .env file should be created in the format below.

        `<key>` and `<string>` should be filled in with the Azure Storage Connection String and the Azure CosmosDB Primary Key above.

        ```
        AZURE_COSMOSDB_PRIMARY_KEY=<key>
        AZURE_STORAGE_CONNECTION_STRING=<string>
        INFERENCESYSTEM_APPINSIGHTS_CONNECTION_STRING=<string>
        ```

## Adding a new hydrophone

1. Create a new config file under the [config](config) folder

2. Update the last line of the [Dockerfile](Dockerfile) to point to the new config file 

3. Create a new deployment YAML under the [deploy](deploy) folder

4. Update [src/LiveInferenceOrchestrator.py](src/LiveInferenceOrchestrator.py) and [src/globals.py](src/globals.py) to add variables for the new hydrophone location 

5. Follow all other steps below until you update the kubernetes cluster with the new namespace

## Building the docker container for production

From the `InferenceSystem` directory, run the following command.
It will take a while (~2-3 minutes on macOS or Linux, ~10-20 minutes on Windows) the first time, but builds are cached, and it
should take a much shorter time in future builds.

```
docker build . -t live-inference-system -f ./Dockerfile
```

Note: the config used in the Dockerfile is a Production config.

TODO: fix. For now, you will have to manually create 5 different docker containers for the 5 hydrophone locations. Each time you will need to edit the Dockerfile and replace the config for each hydrophone location (OrcasoundLab, BushPoint, PortTownsend, Sunset Bay and Point Robinson).


## Running the docker container

From the `InferenceSystem` directory, run the following command.

```
docker run --rm -it --env-file .env live-inference-system
```

In addition, you should see something similar to the following in your console.

```
Listening to location https://s3-us-west-2.amazonaws.com/streaming-orcasound-net/rpi_orcasound_lab
Downloading live879.ts
live879.ts: 205kB [00:00, 1.17MB/s]                                             
Downloading live880.ts
live880.ts: 205kB [00:00, 1.11MB/s]                                             
Downloading live881.ts
live881.ts: 205kB [00:00, 948kB/s]                                              
Downloading live882.ts
live882.ts: 205kB [00:00, 1.14MB/s]                                             
Downloading live883.ts
live883.ts: 205kB [00:00, 1.07MB/s]                                             
Downloading live884.ts
live884.ts: 205kB [00:00, 1.04MB/s]                                             
rpi_orcasound_lab_2021_10_13_15_11_18_PDT.wav
Length of Audio Clip:60.010666666666665
Preprocessing: Downmixing to Mono
Preprocessing: Resampling to 200009/59 00:00<00:00]
```

# Pushing your image to Azure Container Registry

This step pushes your local container to the Azure Container Registry (ACR).  If you would like more information, this
documentation is adapted from 
[this tutorial](https://docs.microsoft.com/en-us/azure/container-instances/container-instances-tutorial-prepare-acr).

1. Login to the shared azure directory from the Azure CLI.

```
az login --tenant adminorcasound.onmicrosoft.com
```

2. We will be using the orcaconservancycr ACR in the LiveSRKWNotificationSystem Resource Group. Log in to the container registry.

```
az acr login --name orcaconservancycr
```

You should receive something similar to `Login succeeded`.

3. Tag your docker container with the version number. We use the following versioning scheme.

```
docker tag live-inference-system orcaconservancycr.azurecr.io/live-inference-system:<date-of-deployment>.<model-type>.<Rounds-trained-on>.<hydrophone-location>.v<Major>
```

So, for example your command may look like

```
docker tag live-inference-system orcaconservancycr.azurecr.io/live-inference-system:11-15-20.FastAI.R1-12.OrcasoundLab.v0
```

Look at [deploy-aci.yaml](deploy-aci.yaml) for examples of how previous models were tagged.

4. Lastly, push your image to Azure Container Registry for each Orcasound Hydrophone Location.

```
docker push orcaconservancycr.azurecr.io/live-inference-system:<date-of-deployment>.<model-type>.<Rounds-trained-on>.<hydrophone-location>.v<Major>
```

# Deploying an updated docker build to Azure Kubernetes Service

We are deploying one hydrophone per namespace. To deploy a hydrophone, the following Kubernetes resources need to be created:

1. Namespace: used to group resources
2. Secret: holds connection strings used by inference system
3. Deployment: forces one instance of inference system to remain running at all times

## Prerequisites

- You must have completed all of the steps above and should have a working container image pushed to ACR.
- az cli: installation instructions [here](https://learn.microsoft.com/en-us/cli/azure/install-azure-cli)
- kubectl cli: if you don't have this, you can run `az aks install-cli` or install it using instructions [here](https://kubernetes.io/docs/tasks/tools/)

1. Log into az cli

```bash
az login
```

2. Log into Kubernetes cluster. The current cluster is called inference-system-AKS in the LiveSRKWNotificationSystem resource group.

```bash
# replace "inference-system-AKS" with cluster name and "LiveSRKWNotificationSystem" with resource group
az aks get-credentials -g LiveSRKWNotificationSystem -n inference-system-AKS
```

Verify it is successful. You should see a list of VM names and no error message.

```bash
kubectl get nodes
```

3. If deploying a new hydrophone, create a new namespace and secret. Skip this step if not bringing up a new hydrophone.

```bash
# replace "bush-point" with hydrophone identifier
kubectl create namespace bush-point

kubectl create secret generic inference-system -n bush-point \
    --from-literal=AZURE_COSMOSDB_PRIMARY_KEY='<cosmos_primary_key>' \
    --from-literal=AZURE_STORAGE_CONNECTION_STRING='<storage_connection_string>`' \
    --from-literal=INFERENCESYSTEM_APPINSIGHTS_CONNECTION_STRING='<appinsights_connection_string>'
```

4. Create or update deployment. Use file for hydrophone under [deploy](./deploy/) folder, or create and commit a new one.

```bash
kubectl apply -f deploy/bush-point.yaml
```

5. To verify that the container is running, check logs:

```bash
# get pod name
kubectl get pods -n bush-point

# replace pod name (likely will have different alphanumeric string at the end)
kubectl logs -n bush-point inference-system-6d4845c5bc-tfsbw
```

<details>
  <summary>Deployment to Azure Container Instances (deprecated)</summary>
# Deploying an updated docker build to Azure Container Instances
# This method has been deprecated

## Prerequisites

- You must have completed all of the steps above and should have a 
container that is working locally that you wish to deploy live to production.

- **Azure CLI**: You must have Azure CLI version 2.0.29 or later installed on your local computer. Run `az --version` to find the 
version. If you need to install or upgrade, see 
[Install the Azure CLI](https://docs.microsoft.com/en-us/cli/azure/install-azure-cli).

## Deploying your updated container to Azure Container Instances

Ask an existing maintainer for the file `deploy-aci-with-creds.yaml` or change strings in `deploy-aci.yaml`.  There are three sensitive strings that must be filled in before deployment can
happen.

**NOTE** - Make sure you change these back after running the build - don't commit them to the repository!

1.  `<cosmos_primary_key>` - Replace this with the AZURE_COSMOSDB_PRIMARY_KEY from your .env file (or found above).
2.  `<storage_connection_string>` - Replace this with the AZURE_STORAGE_CONNECTION_STRING from your .env file (or found above).
3.  `<appinsights_connection_string>` - Replace this with the INFERENCESYSTEM_APPINSIGHTS_CONNECTION_STRING from your .env file (or found above).
4.  `<image_registry_password>` - Replace this with the password for the orcaconservancycr container registry.  It can be found at
[this link](https://portal.azure.com/#@OrcaConservancy778.onmicrosoft.com/resource/subscriptions/9ffa543e-3596-43aa-b82c-8f41dfbf03cc/resourcegroups/LiveSRKWNotificationSystem/providers/Microsoft.ContainerRegistry/registries/orcaconservancycr/accessKey)
under the name `password`.

Then, run this command from the `InferenceSystem` directory.  It will take a while to complete.  Once complete, make sure to check your work below.

```
az container create -g LiveSRKWNotificationSystem -f .\deploy-aci.yaml
```

## Checking your work

View the container logs with the following command.  The logs should be similar to the logs created when you run the container locally (above).

```
az container attach --resource-group LiveSRKWNotificationSystem --name live-inference-system-aci-3gb-new
```

</details>

# No changes made to deploy-aci.yaml?

I purposefully told git to ignore all futher changes to the file with this command: `git update-index --assume-unchanged deploy-aci.yaml`.  This is to prevent people from checking in their credentials into the repository.  If you want a change to be tracked, you can turn off this feature with `git update-index --no-assume-unchanged deploy-aci.yaml`


# Running the automatic annotation data upload script PrepareDataForPredictionExplorer.py

This script processes audio from a segment of data and uploads it to the annotation website [https://aifororcas-podcast.azurewebsites.net/](https://aifororcas-podcast.azurewebsites.net/).

To run the script, find the connection string for blob storage account `"mldevdatastorage"` and run the following 

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

Call the script as follows, substituting appropriate values.

```
python PrepareDataForPredictionExplorer.py --start_time "2020-07-25 19:15" --end_time "2020-07-25 20:15" --s3_stream https://s3-us-west-2.amazonaws.com/streaming-orcasound-net/rpi_orcasound_lab --model_path <folder> --annotation_threshold 0.4 --round_id round5 --dataset_folder <path-to-folder>
```
