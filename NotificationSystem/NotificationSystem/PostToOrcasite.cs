using ComposableAsync;
using Microsoft.Azure.Functions.Worker;
using Microsoft.Extensions.Logging;
using NotificationSystem.Models;
using RateLimiter;
using System;
using System.Collections.Generic;
using System.Diagnostics.Eventing.Reader;
using System.Text.Json;
using System.Threading;
using System.Threading.Tasks;

namespace NotificationSystem
{
    public class PostToOrcasite
    {
        private readonly OrcasiteHelper _helper;
        private readonly ILogger _logger;
        const int SendRate = 14; // Max 14 posts per second.

        public PostToOrcasite(OrcasiteHelper helper, ILogger<PostToOrcasite> logger)
        {
            _helper = helper;
            _logger = logger;
        }

        public async Task<bool> ProcessDocumentsAsync(
            IReadOnlyList<JsonElement> input)
        {
            if (input == null || input.Count == 0)
            {
                _logger.LogInformation("No updated records");
                return true;
            }

            await _helper.InitializeAsync();

            var timeConstraint = TimeLimiter.GetFromMaxCountByInterval(SendRate, TimeSpan.FromSeconds(1));
            int remaining = input.Count;

            foreach (var document in input)
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

        [Function("PostToOrcasite")]
        public async Task Run(
            [CosmosDBTrigger(
                databaseName: "predictions",
                containerName: "metadata",
                Connection = "aifororcasmetadatastore_DOCUMENTDB",
                LeaseContainerName = "leases",
                LeaseContainerPrefix = "orcasite",
                CreateLeaseContainerIfNotExists = true)] IReadOnlyList<JsonElement> input)
        {
            // Currently all data in the OrcaHello database has incorrect timestamps.
            // We correct for that here.
            // TODO(issue #219): Remove this workaround when the data is fixed.
            IReadOnlyList<JsonElement> correctedInput = await _helper.FixTimestampsAsync(input);

            bool ok = await ProcessDocumentsAsync(correctedInput);
            if (ok)
            {
                Interlocked.Increment(ref _successfulRuns);
            }
        }
    }
}
