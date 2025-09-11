using Azure.Data.Tables;
using Microsoft.Azure.Functions.Worker;
using Microsoft.Azure.Functions.Worker.Http;
using Microsoft.Extensions.Logging;
using NotificationSystem.Models;
using NotificationSystem.Utilities;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Net;
using System.Threading.Tasks;

namespace NotificationSystem
{
    public class SubscriberEmailFunctions
    {
        private readonly ILogger _logger;
        public SubscriberEmailFunctions(ILogger<SubscriberEmailFunctions> logger)
        {
            _logger = logger;
        }

        [Function("SubscribeToSubscriberEmail")]
        public async Task<HttpResponseData> Subscribe(
            [HttpTrigger(AuthorizationLevel.Function, "post", "delete")] HttpRequestData req,
            [TableInput("EmailList", Connection = "OrcaNotificationStorageSetting")] TableClient tableClient,
            FunctionContext context)
        {
            _logger.LogInformation("Processing subscription request for subscriber email.");

            try
            {
                var response = await EmailHelpers.UpdateEmailListAsync<SubscriberEmailEntity>(req, tableClient, _logger);
                return response;
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Failed to update subscriber email list.");
                var errorResponse = req.CreateResponse(HttpStatusCode.InternalServerError);
                await errorResponse.WriteStringAsync("An error occurred while processing your request.");
                return errorResponse;
            }
        }

        [Function("ListSubscriberEmails")]
        public async Task<HttpResponseData> List(
            [HttpTrigger(AuthorizationLevel.Function, "get")] HttpRequestData req,
            [TableInput("EmailList", Connection = "OrcaNotificationStorageSetting")] TableClient tableClient)
        {
            var emails = await EmailHelpers.GetEmailEntitiesAsync<SubscriberEmailEntity>(tableClient, "Subscriber");
            var response = req.CreateResponse(HttpStatusCode.OK);
            await response.WriteAsJsonAsync(emails.Select(e => e.Email));
            return response;
        }
    }
}
