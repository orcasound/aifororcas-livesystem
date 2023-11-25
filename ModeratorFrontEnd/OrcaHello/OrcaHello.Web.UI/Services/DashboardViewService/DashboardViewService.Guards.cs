namespace OrcaHello.Web.UI.Services
{
    /// <summary>
    /// Partial of the the <see cref="DashboardViewService"/> orchestration service class responsible for peforming 
    /// level-specific validations.
    /// </summary>
    public partial class DashboardViewService
    {
        // RULE: Date range must be valid.
        // It checks if the parameters are valid and throws an InvalidDashboardException if not.
        private void ValidateDateRange(DateTime? fromDate, DateTime? toDate)
        {
            // If the fromDate parameter is not null and is greater than the current date, throw an InvalidDashboardViewException with a custom message.
            if (fromDate.HasValue && fromDate.Value > DateTime.Now)
            {
                throw new InvalidDashboardViewException("The from date cannot be in the future.");
            }

            // If the toDate parameter is not null and is less than the fromDate parameter, throw an InvalidDashboardViewException with a custom message.
            if (toDate.HasValue && toDate.Value < fromDate)
            {
                throw new InvalidDashboardViewException("The to date cannot be before the from date.");
            }
        }

        // RULE: Pagination must be correct.
        // It checks if the parameters are positive and throws an InvalidDashboardViewException if not.
        private void ValidatePagination(int page, int pageSize)
        {
            // If the page parameter is less than or equal to zero, throw an InvalidDashboardViewException with a custom message.
            if (page <= 0)
            {
                throw new InvalidDashboardViewException("The page number must be positive.");
            }

            // If the pageSize parameter is less than or equal to zero, throw an InvalidDashboardViewException with a custom message.
            if (pageSize <= 0)
            {
                throw new InvalidDashboardViewException("The page size must be positive.");
            }
        }

        // RULE: Check if the moderator name is valid.
        // It checks if the parameter is null, empty, or whitespace and throws an InvalidDashboardViewException if so.
        private void ValidateModerator(string moderator)
        {
            // If the moderator parameter is null, empty, or whitespace, throw an InvalidDashboardViewException with a custom message.
            if (String.IsNullOrWhiteSpace(moderator))
            {
                throw new InvalidDashboardViewException("The moderator name cannot be null, empty, or whitespace.");
            }
        }

        // RULE: Check if the tag is valid.
        // It checks if the parameter is null, empty, or whitespace and throws an InvalidDashboardViewException if so.
        private void ValidateTag(string tag)
        {
            // If the tag parameter is null, empty, or whitespace, throw an InvalidDashboardViewException with a custom message.
            if (String.IsNullOrWhiteSpace(tag))
            {
                throw new InvalidDashboardViewException("The tag cannot be null, empty, or whitespace.");
            }
        }

        // RULE: Response cannot be null.
        // It checks if the response is null and throws a NullDashboardViewResponseException if so.
        private static void ValidateResponse<T>(T response)
        {
            // If the response is null, throw a NullDashboardViewResponseException.
            if (response == null)
            {
                throw new NullDashboardViewResponseException();
            }
        }

        // RULE: Request cannot be null.
        // It checks if the request is null and throws a NullDashboardViewRequestException if so.
        private static void ValidateRequest<T>(T request)
        {
            // If the request is null, throw a NullDashboardViewRequestException.
            if (request == null)
            {
                throw new NullDashboardViewRequestException();
            }
        }
    }
}
