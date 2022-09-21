using System;
using System.Collections.Generic;
using System.Threading.Tasks;
using Microsoft.Azure.Cosmos.Table;
using Microsoft.Azure.Documents;
using Microsoft.Azure.WebJobs;
using Microsoft.Extensions.Logging;
using NotificationSystem.Template;
using NotificationSystem.Utilities;
using SendGrid.Helpers.Mail;

namespace NotificationSystem
{
    [StorageAccount("OrcaNotificationStorageSetting")]
    public static class SendModeratorEmail
    {
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
            [SendGrid(ApiKey = "SendGridKey")] IAsyncCollector<SendGridMessage> messageCollector,
            ILogger log)
        {
            if (input == null || input.Count == 0)
            {
                log.LogInformation("No updated records");
                return;
            }

            var newDocumentCreated = false;
            DateTime? documentTimeStamp = null;

            foreach (var document in input)
            {
                if (!document.GetPropertyValue<bool>("reviewed"))
                {
                    newDocumentCreated = true;
                    documentTimeStamp = document.GetPropertyValue<DateTime>("timestamp");
                    break;
                }
            }

            if (!newDocumentCreated)
            {
                log.LogInformation("No unreviewed records");
                return;
            }

            // TODO: make better email
            string body = EmailTemplate.GetModeratorEmailBody(documentTimeStamp);

            log.LogInformation("Retrieving email list and sending notifications");
            foreach (var emailEntity in EmailHelpers.GetEmailEntities(cloudTable, "Moderator"))
            {
                var email = EmailHelpers.CreateEmail(Environment.GetEnvironmentVariable("Please validate new OrcaHello detection"),
                    emailEntity.Email, "New Orca Call Identified", body);
                await messageCollector.AddAsync(email);
            }
        }
    }
}
