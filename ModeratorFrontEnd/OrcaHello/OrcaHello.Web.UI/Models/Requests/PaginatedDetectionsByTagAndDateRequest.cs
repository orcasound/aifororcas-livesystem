namespace OrcaHello.Web.UI.Models
{
    [ExcludeFromCodeCoverage]
    public class PaginatedDetectionsByTagAndDateRequest : DateRequestBase
    {
        public string Tag { get; set; } = null!;
        public int Page { get; set; }
        public int PageSize { get; set; }
    }
}
