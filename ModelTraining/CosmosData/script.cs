using System;
using Microsoft.Azure.Cosmos;

string endpoint = "https://aifororcasmetadatastore.documents.azure.com:443/";
string key = "[INSERT PRIMARY KEY HERE]";

CosmosClient client = new CosmosClient(endpoint, key);
AccountProperties account = await client.ReadAccountAsync();

// Sanity check
Console.WriteLine($"Account Name:\t{account.Id}");

// Get the database
Database database = await client.CreateDatabaseIfNotExistsAsync("predictions");
Container container = await database.CreateContainerIfNotExistsAsync("metadata", "/source_guid");

string sql = "SELECT m.id, m.modelId, m.audioUri, m.imageUri, m.reviewed, m.timestamp, m.whaleFoundConfidence, m.location.id AS location_id, m.location.name AS location_name, m.location.longitude AS location_long, m.location.latitude AS location_lat, m.source_guid, p.id AS prediction_id, p.startTime AS prediction_startTime, p.duration AS prediction_duration, p.confidence AS prediction_confidence FROM metadata m JOIN p IN m.predictions";
QueryDefinition query = new (sql);

QueryRequestOptions options = new ();
options.MaxItemCount = 50;

FeedIterator<Prediction> iterator = container.GetItemQueryIterator<Prediction>(query, requestOptions: options);

while (iterator.HasMoreResults)
{
    FeedResponse<Prediction> predictions = await iterator.ReadNextAsync();
    foreach (Prediction pred in predictions)
    {
        Console.WriteLine($"[{pred.prediction_id}]\t[{pred.prediction_startTime,40}]\t[{pred.prediction_duration,10}]\t[{pred.prediction_duration,40}]\t[{pred.prediction_confidence,40}]\t");
    }
Console.WriteLine("Press any key for next page of results");
Console.ReadKey();        
}

