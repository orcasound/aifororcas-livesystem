using System;
using Amazon;
using Amazon.SimpleEmail;
using RateLimiter;
using ComposableAsync;
using System.Collections.Generic;
using System.Threading.Tasks;
using Microsoft.Azure.Cosmos.Table;
using Microsoft.Azure.Documents;
using Microsoft.Azure.WebJobs;
using Microsoft.Extensions.Logging;
using NotificationSystem.Template;
using NotificationSystem.Utilities;

namespace NotificationSystem
{
    [StorageAccount("OrcaNotificationStorageSetting")]
    public static class SendModeratorEmail
    {
        static int SendRate = 14;

        [FunctionName("SendModeratorEmail")]
        public static async Task Run(
            [CosmosDBTrigger(
                databaseName: "predictions",
                collectionName: "metadata",
                ConnectionStringSetting = "aifororcasmetadatastore_DOCUMENTDB",
                LeaseCollectionName = "leases",
                LeaseCollectionPrefix = "moderator",
                CreateLeaseCollectionIfNotExists = true)]IReadOnlyList<Document> input,
            [Table("EmailList")] CloudTable cloudTable,
            ILogger log)
        {
            if (input == null || input.Count == 0)
            {
                log.LogInformation("No updated records");
                return;
            }

            var newDocumentCreated = false;
            DateTime? documentTimeStamp = null;
            string location = null;

            foreach (var document in input)
            {
                if (!document.GetPropertyValue<bool>("reviewed"))
                {
                    newDocumentCreated = true;
                    documentTimeStamp = document.GetPropertyValue<DateTime>("timestamp");
                    location = document.GetPropertyValue<string>("location.name");
                    break;
                }
            }

            if (!newDocumentCreated)
            {
                log.LogInformation("No unreviewed records");
                return;
            }

            string body = EmailTemplate.GetModeratorEmailBody(documentTimeStamp, location);

            var timeConstraint = TimeLimiter.GetFromMaxCountByInterval(SendRate, TimeSpan.FromSeconds(1));
            var aws = new AmazonSimpleEmailServiceClient(RegionEndpoint.USWest2);
            log.LogInformation("Retrieving email list and sending notifications");
            foreach (var emailEntity in EmailHelpers.GetEmailEntities(cloudTable, "Moderator"))
            {
                await timeConstraint;
				string emailSubject = string.Format("OrcaHello Candidate at location {0}", location);
                var email = EmailHelpers.CreateEmail(Environment.GetEnvironmentVariable("SenderEmail"),
                    emailEntity.Email, emailSubject, body);
                await aws.SendEmailAsync(email);
            }
        }
    }
}
