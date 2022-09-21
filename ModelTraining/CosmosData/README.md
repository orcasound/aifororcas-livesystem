# Extracting additional training data from CosmosDB

An example of an item the metadata store appears as follows in our CosmosDB database: 

```
{
    "id": "1ed0f937-3f63-4f6a-a680-1b4b4ef24fb9",
    "modelId": "AudioSet",
    "audioUri": "https://livemlaudiospecstorage.blob.core.windows.net/audiowavs/rpi_orcasound_lab_2020_09_27_21_15_03_PDT.wav",
    "imageUri": "https://livemlaudiospecstorage.blob.core.windows.net/spectrogramspng/rpi_orcasound_lab_2020_09_27_21_15_03_PDT.png",
    "reviewed": true,
    "timestamp": "2020-09-28T04:15:03.495901Z",
    "whaleFoundConfidence": 80.18461538461537,
    "location": {
        "id": "rpi_orcasound_lab",
        "name": "Haro Strait",
        "longitude": -123.2166658,
        "latitude": 48.5499978
    },
    "source_guid": "rpi_orcasound_lab",
    "predictions": [
        {
            "id": 0,
            "startTime": 2.5,
            "duration": 2.5,
            "confidence": 0.914
        },
        {
            "id": 1,
            "startTime": 7.5,
            "duration": 2.5,
            "confidence": 0.624
        },
        {
            "id": 2,
            "startTime": 12.5,
            "duration": 2.5,
            "confidence": 0.869
        },
        {
            "id": 3,
            "startTime": 15,
            "duration": 2.5,
            "confidence": 0.918
        }
    ]
}
```

Attached is a .NET application that allows you to create a cross-product of each observation to the predictions property, resulting in a JSON array of all the possible permutations between the relevant observation metadata and each unique prediction. 

The steps required to leverage this .NET application is detailed in this [link](https://learn.microsoft.com/en-us/training/paths/connect-to-azure-cosmos-db-sql-api-sdk/). It requires downloading the `Microsoft.Azure.Cosmos` package from `nuget.org`, connecting to our online account and executing the SQL query as specified in the `script.cs`. To build this application, please run `dotnet run`. 