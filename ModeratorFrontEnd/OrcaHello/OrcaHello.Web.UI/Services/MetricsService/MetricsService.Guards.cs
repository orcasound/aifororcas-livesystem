namespace OrcaHello.Web.UI.Services
{
    /// <summary>
    /// Partial of the the <see cref="MetricsService"/> foundation service class responsible for peforming 
    /// level-specific validations.
    /// </summary>
    public partial class MetricsService
    {
        // RULE: Date range must be valid.
        // It checks if the parameters are valid and throws an InvalidMetricsException if not.
        private void ValidateDateRange(DateTime? fromDate, DateTime? toDate)
        {
            // If the fromDate parameter is not null and is greater than the current date, throw an InvalidMetricsException with a custom message.
            if (fromDate.HasValue && fromDate.Value > DateTime.Now)
                throw new InvalidMetricsException("The from date cannot be in the future.");

            // If the toDate parameter is not null and is less than the fromDate parameter, throw an InvalidMetricsException with a custom message.
            if (toDate.HasValue && toDate.Value < fromDate)
                throw new InvalidMetricsException("The to date cannot be before the from date.");
        }

        // RULE: Response cannot be null.
        // It checks if the response is null and throws a NullMetricsResponseException if so.
        private static void ValidateResponse<T>(T response)
        {
            // If the response is null, throw a NullMetricsResponseException.
            if (response == null)
            {
                throw new NullMetricsResponseException();
            }
        }
    }
}
