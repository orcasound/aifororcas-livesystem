using System;
using System.Collections.Generic;
using Microsoft.Azure.Documents;
using Microsoft.Azure.WebJobs;
using Microsoft.Extensions.Logging;
using Azure.Storage.Queues;
using NotificationSystem.Utilities;

namespace NotificationSystem
{ 
    public static class DbToQueue
    {
        [FunctionName("DbToQueue")]
        public static void Run([CosmosDBTrigger(
            databaseName: "predictions",
            collectionName: "metadata",
            ConnectionStringSetting = "aifororcasmetadatastore_DOCUMENTDB",
            LeaseCollectionName = "leases",
            LeaseCollectionPrefix = "subscriber",
            CreateLeaseCollectionIfNotExists = true)]IReadOnlyList<Document> input, ILogger log)
        {
            // Instantiate a QueueClient which will be used to create and manipulate the queue
            string connectionString = Environment.GetEnvironmentVariable("OrcaNotificationStorageSetting");
            QueueClient queueClient = new QueueClient(connectionString, "srkwfound");

            if (input != null && input.Count > 0)
            {
                // Send a message to the queue
                int numberSent = 0;
                foreach (var item in input)
                {
                    if (item.GetPropertyValue<string>("SRKWFound").ToLower() == "yes")
                    {
                        queueClient.SendMessage(StringHelpers.Base64Encode(item.ToString()));
                        log.LogInformation($"Document with id {item.Id} sent");
                        numberSent++;
                    }
                }

                log.LogInformation($"Number of initial documents : {input.Count}");
                log.LogInformation($"Number of documents added: {numberSent}");
            }
        }
    }
}
