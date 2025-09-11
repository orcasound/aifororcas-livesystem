using Azure.Data.Tables;
using Microsoft.Azure.Functions.Worker;
using Microsoft.Azure.Functions.Worker.Extensions.Tables;
using Microsoft.Azure.Functions.Worker.Http;
using Microsoft.Extensions.Logging;
using NotificationSystem.Models;
using NotificationSystem.Utilities;
using System.Linq;
using System.Net;
using System.Threading.Tasks;

namespace NotificationSystem
{
    public class ModeratorEmailFunctions
    {
        private readonly ILogger _logger;

        public ModeratorEmailFunctions(ILogger<ModeratorEmailFunctions> logger)
        {
            _logger = logger;
        }

        [Function("SubscribeToModeratorEmail")]
        public async Task<HttpResponseData> Subscribe(
            [HttpTrigger(AuthorizationLevel.Function, "post", "delete")] HttpRequestData req,
            [TableInput("EmailList", Connection = "OrcaNotificationStorageSetting")] TableClient tableClient,
            FunctionContext context)
        {
            var response = await EmailHelpers.UpdateEmailListAsync<ModeratorEmailEntity>(req, tableClient, _logger);
            return response;
        }

        [Function("ListModeratorEmails")]
        public async Task<HttpResponseData> List(
            [HttpTrigger(AuthorizationLevel.Function, "get")] HttpRequestData req,
            [TableInput("EmailList", Connection = "OrcaNotificationStorageSetting")] TableClient tableClient,
            FunctionContext context)
        {
            var emails = await EmailHelpers.GetEmailEntitiesAsync<ModeratorEmailEntity>(tableClient, "Moderator");
            var response = req.CreateResponse(HttpStatusCode.OK);
            await response.WriteAsJsonAsync(emails.Select(e => e.Email));
            return response;
        }
    }
}
