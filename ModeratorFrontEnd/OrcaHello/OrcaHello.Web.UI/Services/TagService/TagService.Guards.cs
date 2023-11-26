namespace OrcaHello.Web.UI.Services
{
    /// <summary>
    /// Partial of the the <see cref="TagService"/> foundation service class responsible for peforming 
    /// level-specific validations.
    /// </summary>
    public partial class TagService
    {
        // RULE: String property must be present.
        protected void Validate(string propertyValue, string propertyName)
        {
            if (ValidatorUtilities.IsInvalid(propertyValue))
                throw new InvalidDetectionException(LoggingUtilities.MissingRequiredProperty(propertyName));
        }

        // RULE: Date range must be valid.
        protected void ValidateDateRange(DateTime? fromDate, DateTime? toDate)
        {
            if (fromDate.HasValue && fromDate.Value > DateTime.UtcNow)
                throw new InvalidTagException("Property 'fromDate' cannot be in the future.");

            if (toDate.HasValue && toDate.Value < fromDate)
                throw new InvalidTagException("Property 'toDate' cannot be before the 'fromDate'.");
        }

        // RULE: Request cannot be null.
        protected static void ValidateRequest<T>(T response)
        {
            if (response == null)
            {
                throw new NullTagRequestException();
            }
        }

        // RULE: Response cannot be null.
        protected static void ValidateResponse<T>(T response)
        {
            if (response == null)
            {
                throw new NullTagResponseException();
            }
        }
    }
}
