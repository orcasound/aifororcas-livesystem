using Amazon;
using Amazon.SimpleEmail;
using Azure.Data.Tables;
using Azure.Storage.Queues;
using Azure.Storage.Queues.Models;
using ComposableAsync;
using Microsoft.Azure.Functions.Worker;
using Microsoft.Extensions.Configuration;
using Microsoft.Extensions.Logging;
using Newtonsoft.Json;
using Newtonsoft.Json.Linq;
using NotificationSystem.Models;
using NotificationSystem.Template;
using NotificationSystem.Utilities;
using RateLimiter;
using System;
using System.Collections.Generic;
using System.Text;
using System.Threading.Tasks;

namespace NotificationSystem
{
    public class SendSubscriberEmail
    {
        private readonly ILogger _logger;
        private readonly OrcasiteHelper _orcasiteHelper;
        private readonly IConfiguration _configuration;
        const int SendRate = 14;

        public SendSubscriberEmail(ILogger<SendSubscriberEmail> logger, OrcasiteHelper orcasiteHelper, IConfiguration configuration)
        {
            _logger = logger;
            _orcasiteHelper = orcasiteHelper;
            _configuration = configuration;
        }

        [Function("SendSubscriberEmail")]
        // TODO: change timer to once per hour (0 0 * * * *)
        public async Task Run(
            [TimerTrigger("0 */1 * * * *")] string timerInfo,
            [TableInput("EmailList", Connection = "OrcaNotificationStorageSetting")] TableClient tableClient)
        {
            // Initialize OrcasiteHelper to fetch feeds data
            await _orcasiteHelper.InitializeAsync(_configuration);
            
            string queueConnection = Environment.GetEnvironmentVariable("OrcaNotificationStorageSetting");
            var queueClient = new QueueClient(queueConnection, "srkwfound");

            _logger.LogInformation("Checking if there are items in queue");
            QueueProperties properties = await queueClient.GetPropertiesAsync();

            if (properties.ApproximateMessagesCount == 0)
            {
                _logger.LogInformation("No items in queue");
                return;
            }

            _logger.LogInformation("Creating email message");
            var (body, messages) = await CreateBody(queueClient);

            var timeConstraint = TimeLimiter.GetFromMaxCountByInterval(SendRate, TimeSpan.FromSeconds(1));
            var aws = new AmazonSimpleEmailServiceClient(RegionEndpoint.USWest2);
            _logger.LogInformation("Retrieving email list and sending notifications");
            string emailSubject = EmailTemplate.GetSubscriberEmailSubject(messages);
            foreach (var emailEntity in await EmailHelpers.GetEmailEntitiesAsync<SubscriberEmailEntity>(tableClient, "Subscriber"))
            {
                await timeConstraint;
                var email = EmailHelpers.CreateEmail(
                    Environment.GetEnvironmentVariable("SenderEmail"),
                    emailEntity.Email,
                    emailSubject,
                    body);
                await aws.SendEmailAsync(email);
            }
        }

        private async Task<(string body, List<JObject> messages)> CreateBody(QueueClient queueClient)
        {
            var bodyBuilder = new StringBuilder("<h1>Confirmed SRKW detections:</h1>\n<ul>");
            QueueMessage message;
            List<JObject> messagesJson = new List<JObject>();

            while (true)
            {
                var response = await queueClient.ReceiveMessageAsync();
                message = response.Value;
                if (message == null || string.IsNullOrEmpty(message.MessageText))
                    break;

                var decoded = Encoding.UTF8.GetString(Convert.FromBase64String(message.MessageText));
                messagesJson.Add(JsonConvert.DeserializeObject<JObject>(decoded));
                await queueClient.DeleteMessageAsync(message.MessageId, message.PopReceipt);
            }

            string body = EmailTemplate.GetSubscriberEmailBody(messagesJson, _orcasiteHelper);
            return (body, messagesJson);
        }
    }
}