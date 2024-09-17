namespace OrcaHello.Web.Shared.Extensions
{
    [ExcludeFromCodeCoverage]
    public static class ExceptionExtensions
    {
        // Extension method to get all messages from an exception and its inner exceptions
        public static string GetAllMessages(this Exception ex)
        {
            // Use a StringBuilder to append the messages
            var sb = new StringBuilder();

            // Loop through the exception and its inner exceptions
            while (ex != null)
            {
                // Append the current exception message
                sb.AppendLine(ex.Message);

                // Move to the next inner exception
                ex = ex.InnerException;
            }

            // Return the concatenated messages
            return sb.ToString();
        }

        // Extension method to get the stack trace of an exception and its inner exceptions
        public static string GetFullStackTrace(this Exception ex)
        {
            // Use a StringBuilder to append the stack traces
            var sb = new StringBuilder();

            // Loop through the exception and its inner exceptions
            while (ex != null)
            {
                // Append the current exception stack trace
                sb.AppendLine(ex.StackTrace);

                // Move to the next inner exception
                ex = ex.InnerException;
            }

            // Return the concatenated stack traces
            return sb.ToString();
        }
    }
}