namespace OrcaHello.Web.UI.Services
{
    /// <summary>
    /// Partial of the the <see cref="CommentService"/> foundation service class responsible for peforming 
    /// level-specific validations.
    /// </summary>
    public partial class CommentService
    {
        // RULE: Date range must be valid.
        protected void ValidateDateRange(DateTime? fromDate, DateTime? toDate)
        {
            if (fromDate.HasValue && fromDate.Value > DateTime.UtcNow)
                throw new InvalidCommentException("Property 'fromDate' cannot be in the future.");

            if (toDate.HasValue && toDate.Value < fromDate)
                throw new InvalidCommentException("Property 'toDate' cannot be before the 'fromDate'.");
        }

        // RULE: Pagination must be correct.
        protected void ValidatePagination(int page, int pageSize)
        {
            if (page <= 0)
                throw new InvalidCommentException("Property 'page' number must be positive.");

            if (pageSize <= 0)
                throw new InvalidCommentException("Property 'pageSize' number must be positive.");
        }

        // RULE: Response cannot be null.
        protected static void ValidateResponse<T>(T response)
        {
            if (response == null)
            {
                throw new NullCommentResponseException();
            }
        }
    }
}
