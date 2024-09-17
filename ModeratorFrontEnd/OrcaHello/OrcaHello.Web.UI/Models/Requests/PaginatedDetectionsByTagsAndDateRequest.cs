namespace OrcaHello.Web.UI.Models
{
    [ExcludeFromCodeCoverage]
    public class PaginatedDetectionsByTagsAndDateRequest : DateRequestBase
    {
        public List<string> Tags { get; set; } = null!;
        public LogicalOperator Logic { get; set; }
        public int Page { get; set; }
        public int PageSize { get; set; }
    }
}
