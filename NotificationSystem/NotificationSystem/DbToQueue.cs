using Azure.Storage.Queues;
using Microsoft.Azure.Functions.Worker;
using Microsoft.Extensions.Logging;
using NotificationSystem.Utilities;
using System;
using System.Collections.Generic;
using System.Text.Json;
using System.Threading.Tasks;

namespace NotificationSystem
{
    public class DbToQueue
    {
        private readonly ILogger _logger;

        public DbToQueue(ILogger<DbToQueue> logger)
        {
            _logger = logger;
        }

        [Function("DbToQueue")]
        public async Task Run(
            [CosmosDBTrigger(
                databaseName: "predictions",
                containerName: "metadata",
                Connection = "aifororcasmetadatastore_DOCUMENTDB",
                LeaseContainerName = "leases",
                LeaseContainerPrefix = "subscriber",
                CreateLeaseContainerIfNotExists = true)] IReadOnlyList<JsonElement> input)
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
                    if (item.TryGetProperty("SRKWFound", out JsonElement srkwFound) &&
                        srkwFound.GetString()?.ToLower() == "yes")
                    {
                        string id = item.GetProperty("id").GetString();
                        await queueClient.SendMessageAsync(StringHelpers.Base64Encode(item.GetRawText()));
                        _logger.LogInformation($"Document with id {id} sent");
                        numberSent++;
                    }
                }

                _logger.LogInformation($"Number of initial documents: {input.Count}");
                _logger.LogInformation($"Number of documents added: {numberSent}");
            }
        }
    }
}
