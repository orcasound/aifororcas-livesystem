namespace OrcaHello.Web.UI.Models
{
    public class PaginatedDetectionsByStateRequest : DateRequestBase
    {
        public string State { get; set; }
        public string SortBy { get; set; }
        public bool IsDescending { get; set; }
        public int Page { get; set; }
        public int PageSize { get; set; }
        public string Location { get; set; }
    }
}
