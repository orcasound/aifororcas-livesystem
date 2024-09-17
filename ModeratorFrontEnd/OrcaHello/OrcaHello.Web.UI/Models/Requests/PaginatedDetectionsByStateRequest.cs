namespace OrcaHello.Web.UI.Models
{
    [ExcludeFromCodeCoverage]
    public class PaginatedDetectionsByStateRequest : DateRequestBase
    {
        public string State { get; set; } = null!;
        public string SortBy { get; set; } = null!;
        public bool IsDescending { get; set; }
        public int Page { get; set; }
        public int PageSize { get; set; }
        public string Location { get; set; } = null!;
    }
}
