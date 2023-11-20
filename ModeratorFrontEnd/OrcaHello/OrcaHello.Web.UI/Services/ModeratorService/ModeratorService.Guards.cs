namespace OrcaHello.Web.UI.Services
{
    /// <summary>
    /// Partial of the the <see cref="ModeratorService"/> foundation service class responsible for peforming 
    /// level-specific validations.
    /// </summary>
    public partial class ModeratorService
    {
        // RULE: Check if the moderator name is valid.
        // It checks if the parameter is null, empty, or whitespace and throws an InvalidModeratorException if so.
        private void ValidateModerator(string moderator)
        {
            // If the moderator parameter is null, empty, or whitespace, throw an InvalidModeratorException with a custom message.
            if (String.IsNullOrWhiteSpace(moderator))
                throw new InvalidModeratorException("The moderator name cannot be null, empty, or whitespace.");
        }

        // RULE: Check if the tag is valid.
        // It checks if the parameter is null, empty, or whitespace and throws an InvalidModeratorException if so.
        private void ValidateTag(string tag)
        {
            // If the tag parameter is null, empty, or whitespace, throw an InvalidModeratorException with a custom message.
            if (String.IsNullOrWhiteSpace(tag))
                throw new InvalidModeratorException("The tag cannot be null, empty, or whitespace.");
        }

        // RULE: Date range must be valid.
        // It checks if the parameters are valid and throws an InvalidModeratorException if not.
        private void ValidateDateRange(DateTime? fromDate, DateTime? toDate)
        {
            // If the fromDate parameter is not null and is greater than the current date, throw an InvalidModeratorException with a custom message.
            if (fromDate.HasValue && fromDate.Value > DateTime.Now)
                throw new InvalidModeratorException("The from date cannot be in the future.");

            // If the toDate parameter is not null and is less than the fromDate parameter, throw an InvalidModeratorException with a custom message.
            if (toDate.HasValue && toDate.Value < fromDate)
                throw new InvalidModeratorException("The to date cannot be before the from date.");
        }

        // RULE: Pagination must be correct.
        // It checks if the parameters are positive and throws an InvalidModeratorException if not.
        private void ValidatePagination(int page, int pageSize)
        {
            // If the page parameter is less than or equal to zero, throw an InvalidModeratorException with a custom message.
            if (page <= 0)
                throw new InvalidModeratorException("The page number must be positive.");

            // If the pageSize parameter is less than or equal to zero, throw an InvalidModeratorException with a custom message.
            if (pageSize <= 0)
                throw new InvalidModeratorException("The page size must be positive.");
        }

        // RULE: Response cannot be null.
        // It checks if the response is null and throws a NullModeratorResponseException if so.
        private static void ValidateResponse<T>(T response)
        {
            // If the response is null, throw a NullModeratorResponseException.
            if (response == null)
            {
                throw new NullModeratorResponseException();
            }
        }
    }
}
