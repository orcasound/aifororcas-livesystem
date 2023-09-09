namespace OrcaHello.Web.Shared.Utilities
{
    [ExcludeFromCodeCoverage]
    public static class CosmosUtilities
    {
        public static HttpStatusCode GetHttpStatusCode(Exception ex)
        {
            return ((CosmosException)ex).StatusCode;
        }

        public static string GetReason(Exception ex)
        {
            string message = ex.Message; // get the exception message

            var startIndex = message.IndexOf("{");
            var reasonString = message.Substring(startIndex);
            var endIndex = reasonString.IndexOf(")");
            reasonString = reasonString.Substring(0, endIndex);

            JObject reasonObject = JObject.Parse(reasonString);

            string reasonMessage = (string)reasonObject["errors"][0]["message"]; // get the error message as a string

            return $"Cosmos DB error: {reasonMessage} Contact support for resolution.";
        }

        public static string FormatDate(DateTime dateTime)
        {
            return dateTime.ToString("yyyy-MM-ddTHH:mm:ssZ");
        }
    }
}