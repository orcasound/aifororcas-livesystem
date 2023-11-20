namespace OrcaHello.Web.UI.Services
{
    /// <summary>
    /// Partial of the the <see cref="CommentService"/> foundation service class responsible for peforming 
    /// level-specific validations.
    /// </summary>
    public partial class CommentService
    {
        // RULE: Date range must be valid.
        // It checks if the parameters are valid and throws an InvalidCommentException if not.
        private void ValidateDateRange(DateTime? fromDate, DateTime? toDate)
        {
            // If the fromDate parameter is not null and is greater than the current date, throw an InvalidCommentException with a custom message.
            if (fromDate.HasValue && fromDate.Value > DateTime.Now)
                throw new InvalidCommentException("The from date cannot be in the future.");

            // If the toDate parameter is not null and is less than the fromDate parameter, throw an InvalidCommentException with a custom message.
            if (toDate.HasValue && toDate.Value < fromDate)
                throw new InvalidCommentException("The to date cannot be before the from date.");
        }

        // RULE: Pagination must be correct.
        // It checks if the parameters are positive and throws an InvalidCommentException if not.
        private void ValidatePagination(int page, int pageSize)
        {
            // If the page parameter is less than or equal to zero, throw an InvalidCommentException with a custom message.
            if (page <= 0)
                throw new InvalidCommentException("The page number must be positive.");

            // If the pageSize parameter is less than or equal to zero, throw an InvalidCommentException with a custom message.
            if (pageSize <= 0)
                throw new InvalidCommentException("The page size must be positive.");
        }

        // RULE: Response cannot be null.
        // It checks if the response is null and throws a NullCommentResponseException if so.
        private static void ValidateResponse<T>(T response)
        {
            // If the response is null, throw a NullCommentResponseException.
            if (response == null)
            {
                throw new NullCommentResponseException();
            }
        }
    }
}
