namespace OrcaHello.Web.UI.Services
{
    /// <summary>
    /// Partial of the the <see cref="DashboardViewService"/> orchestration service class responsible for peforming 
    /// level-specific validations.
    /// </summary>
    public partial class DashboardViewService
    {
        // RULE: Date range must be valid.
        private void ValidateDateRange(DateTime? fromDate, DateTime? toDate)
        {
            if (fromDate.HasValue && fromDate.Value > DateTime.UtcNow)
                throw new InvalidDashboardViewException("The 'Start' date cannot be in the future.");

            if (toDate.HasValue && toDate.Value < fromDate)
                throw new InvalidDashboardViewException("The 'End' date cannot be before the 'Start' date.");
        }

        // RULE: Pagination must be correct.
        private void ValidatePagination(int page, int pageSize)
        {
            if (page <= 0)
                throw new InvalidDashboardViewException("The page number must be positive.");

            // If the pageSize parameter is less than or equal to zero, throw an InvalidDashboardViewException with a custom message.
            if (pageSize <= 0)
                throw new InvalidDashboardViewException("The page size must be positive.");
        }

        // RULE: Check if the moderator name is valid.
        private void ValidateModerator(string moderator)
        {
            if (String.IsNullOrWhiteSpace(moderator))
                throw new InvalidDashboardViewException("The moderator name cannot be null, empty, or whitespace.");
        }

        // RULE: Check if the tag is valid.
        private void ValidateTag(string tag)
        {
            if (String.IsNullOrWhiteSpace(tag))
                throw new InvalidDashboardViewException("The tag cannot be null, empty, or whitespace.");
        }

        // RULE: Response cannot be null.
        private static void ValidateResponse<T>(T response)
        {
             if (response == null)
            {
                throw new NullDashboardViewResponseException(nameof(T));
            }
        }

        // RULE: Request cannot be null.
        private static void ValidateRequest<T>(T request)
        {
            if (request == null)
            {
                throw new NullDashboardViewRequestException(nameof(T));
            }
        }
    }
}
