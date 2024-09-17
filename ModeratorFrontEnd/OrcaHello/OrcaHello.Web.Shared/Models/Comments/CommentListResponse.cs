namespace OrcaHello.Web.Shared.Models.Comments
{
    [ExcludeFromCodeCoverage]
    [SwaggerSchema("A paginated list of Comments for a given timeframe.")]
    public class CommentListResponse
    {
        [SwaggerSchema("A paginated list of Comments for the given filter information.")]
        public List<Comment> Comments { get; set; }

        [SwaggerSchema("The total number of comments in the list (for pagination).")]
        public int TotalCount { get; set; }

        [SwaggerSchema("The number of detections for this page.")]
        public int Count { get; set; }

        [SwaggerSchema("The starting date of the timeframe.")]
        public DateTime FromDate { get; set; }

        [SwaggerSchema("The ending date of the timeframe.")]
        public DateTime ToDate { get; set; }

        [SwaggerSchema("The requested page number.")]
        public int Page { get; set; }

        [SwaggerSchema("The requested page size.")]
        public int PageSize { get; set; }
    }
}
