using ComposableAsync;
using Microsoft.Azure.Documents;
using Microsoft.Azure.WebJobs;
using Microsoft.Extensions.Logging;
using NotificationSystem.Models;
using RateLimiter;
using System;
using System.Collections.Generic;
using System.Threading;
using System.Threading.Tasks;

namespace NotificationSystem
{
    [StorageAccount("OrcaNotificationStorageSetting")]
    public class PostToOrcasite
    {
        private readonly OrcasiteHelper _helper;

        public PostToOrcasite(OrcasiteHelper helper)
        {
            _helper = helper;
        }

        const int SendRate = 14; // Max 14 posts per second.

        /// <summary>
        /// Process a list of documents from Cosmos DB and post them to Orcasite.
        /// </summary>
        /// <param name="input">List of documents from Cosmos DB</param>
        /// <param name="log"></param>
        /// <returns>true on success, false on failure</returns>
        public async Task<bool> ProcessDocumentsAsync(
            IReadOnlyList<Document> input,
            ILogger log)
        {
            if (input == null || input.Count == 0)
            {
                log.LogInformation("No updated records");
                return true;
            }

            await _helper.InitializeAsync();

            var timeConstraint = TimeLimiter.GetFromMaxCountByInterval(SendRate, TimeSpan.FromSeconds(1));
            int remaining = input.Count;

            foreach (Document document in input)
            {
                bool ok = await _helper.PostDetectionAsync(document.ToString());
                if (!ok)
                {
                    return false;
                }
                remaining--;
                if (remaining > 0)
                {
                    await timeConstraint;
                }
            }
            return true;
        }

        int _successfulRuns = 0;
        public int SuccessfulRuns => _successfulRuns;

        [FunctionName("PostToOrcasite")]
        public async Task Run(
            [CosmosDBTrigger(
                databaseName: "predictions",
                collectionName: "metadata",
                ConnectionStringSetting = "aifororcasmetadatastore_DOCUMENTDB",
                LeaseCollectionName = "leases",
                LeaseCollectionPrefix = "orcasite",
                CreateLeaseCollectionIfNotExists = true)]IReadOnlyList<Document> input,
            ILogger log)
        {
            bool ok = await ProcessDocumentsAsync(input, log);
            if (ok)
            {
                Interlocked.Increment(ref _successfulRuns);
            }
        }
    }
}
