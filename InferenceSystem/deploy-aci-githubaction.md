### Deploy ACI through GitHub Action (Update New AI4Orca Inferencing docker images)

- Ref. Documents
[Configure a GitHub action to create a container instance](https://docs.microsoft.com/en-us/azure/container-instances/container-instances-github-action#create-service-principal-for-azure-authentication)

After the updated AI4Orca ML infernece docker images are pushed to the ACI registry, the following steps are required to update the ACI deployment.


1. __Step 1. Conigure GitHub Secret__

    Configure following GitHub secret. 


| Secret                |Value| Value                                                                                  |
|-----------------------|-----|-----------------------------------------------------------------------------------|
|AZURE_CREDENTIALS |{...}|The entire JSON output from the service principal creation step|
| REGISTRY_LOGIN_SERVER | orcaconservancycr.azurecr.io|The login server name of your registry (all lowercase). Example: myregistry.azurecr.io |
| REGISTRY_USERNAME     | |The clientId from the JSON output from the service principal creation                  |
| REGISTRY_PASSWORD     | |The clientSecret from the JSON output from the service principal creation              |
| RESOURCE_GROUP        | LiveSRKWNotificationSystem |The name of the resource group you used to scope the service principal                 |
|COSMOS_PRIMARY_KEY|<value>|The primary key of the Cosmos DB account.|
|STORAGE_CONNECTION_STRING|<value>|The connection string of the storage account.|

![](img/github-secret-setting.png)


2. __Modify [.github/workflows/RedeployInferenceACI.yml](../.github/workflows/RedeployInferenceACI.yml) file, update the images name and tag of the three ACR docker images.__

    To update the ACI deployment, you need to update the ACI images name and tag in the [.github/workflows/RedeployInferenceACI.yml](../.github/workflows/RedeployInferenceACI.yml) file. Make sure the images name and tag are pointed to the latest images.

```
    ...
    image: ${{ secrets.REGISTRY_LOGIN_SERVER }}/live-inference-system:11-15-20.FastAI.R1-12.OrcasoundLab.v0
    ...
    image: ${{ secrets.REGISTRY_LOGIN_SERVER }}/live-inference-system:11-15-20.FastAI.R1-12.BushPoint.v0
    ...
    image: ${{ secrets.REGISTRY_LOGIN_SERVER }}/live-inference-system:11-15-20.FastAI.R1-12.PortTownsend.v0
```

3. __Make sure file in ./InferenceSystem/configProduction/ is been updated__. 
 The GitHub action will be triggered by any files updated in ./InferenceSystem/configProduction/ folder. To run the GitHub action, you need to update at least one file in the folder.

 ```yaml
 name: GitHub Actions Deploy AI4Orca ML ACI
on:
  # Triggers the workflow on merges only
  push:
    branches: [ main ]
    paths:
      - "InferenceSystem/configProduction/**"
```

4. __Push the changes to the main branch in the repository.__