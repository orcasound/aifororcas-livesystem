using ComposableAsync;
using Microsoft.Azure.Documents;
using Microsoft.Azure.WebJobs;
using Microsoft.Extensions.Logging;
using NotificationSystem.Models;
using RateLimiter;
using System;
using System.Collections.Generic;
using System.Threading.Tasks;

namespace NotificationSystem
{
    [StorageAccount("OrcaNotificationStorageSetting")]
    public static class PostToOrcasite
    {
        static int SendRate = 14; // Max 14 posts per second.

        public static async Task ProcessDocumentsAsync(
            IReadOnlyList<Document> input,
            OrcasiteHelper orcasiteHelper,
            ILogger log)
        {
            if (input == null || input.Count == 0)
            {
                log.LogInformation("No updated records");
                return;
            }

            await orcasiteHelper.InitializeAsync();

            var timeConstraint = TimeLimiter.GetFromMaxCountByInterval(SendRate, TimeSpan.FromSeconds(1));
            int remaining = input.Count;

            foreach (Document document in input)
            {
                await orcasiteHelper.PostDetectionAsync(document.ToString());
                remaining--;
                if (remaining > 0)
                {
                    await timeConstraint;
                }
            }
        }

        [FunctionName("PostToOrcasite")]
        public static async Task Run(
            [CosmosDBTrigger(
                databaseName: "predictions",
                collectionName: "metadata",
                ConnectionStringSetting = "aifororcasmetadatastore_DOCUMENTDB",
                LeaseCollectionName = "leases",
                LeaseCollectionPrefix = "orcasite",
                CreateLeaseCollectionIfNotExists = true)]IReadOnlyList<Document> input,
            ILogger log)
        {
            var orcasiteHelper = new OrcasiteHelper(log);
            await ProcessDocumentsAsync(input, orcasiteHelper, log);
        }
    }
}
