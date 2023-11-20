namespace OrcaHello.Web.UI.Services
{
    /// <summary>
    /// Partial of the the <see cref="TagService"/> foundation service class responsible for peforming 
    /// level-specific validations.
    /// </summary>
    public partial class TagService
    {
        // RULE: Check if the tag is valid.
        // It checks if the parameter is null, empty, or whitespace and throws an InvalidTagException if so.
        private void ValidateTag(string tag)
        {
            // If the tag parameter is null, empty, or whitespace, throw an InvalidTagException with a custom message.
            if (String.IsNullOrWhiteSpace(tag))
                throw new InvalidTagException("The tag cannot be null, empty, or whitespace.");
        }

        // RULE: Date range must be valid.
        // It checks if the parameters are valid and throws an InvalidTagException if not.
        private void ValidateDateRange(DateTime? fromDate, DateTime? toDate)
        {
            // If the fromDate parameter is not null and is greater than the current date, throw an InvalidTagException with a custom message.
            if (fromDate.HasValue && fromDate.Value > DateTime.Now)
                throw new InvalidTagException("The from date cannot be in the future.");

            // If the toDate parameter is not null and is less than the fromDate parameter, throw an InvalidTagException with a custom message.
            if (toDate.HasValue && toDate.Value < fromDate)
                throw new InvalidTagException("The to date cannot be before the from date.");
        }

        // RULE: Request cannot be null and both fields must be populated.
        private static void ValidateReplaceTagRequest(ReplaceTagRequest request)
        {
            if (request is null)
                throw new NullTagRequestException();

            if (String.IsNullOrWhiteSpace(request.OldTag))
                throw new InvalidTagException("The old tag cannot be null, empty, or whitespace.");

            if (String.IsNullOrWhiteSpace(request.NewTag))
                throw new InvalidTagException("The new tag cannot be null, empty, or whitespace.");
        }


        // RULE: Response cannot be null.
        // It checks if the response is null and throws a NullTagResponseException if so.
        private static void ValidateResponse<T>(T response)
        {
            // If the response is null, throw a NullTagResponseException.
            if (response == null)
            {
                throw new NullTagResponseException();
            }
        }
    }
}
