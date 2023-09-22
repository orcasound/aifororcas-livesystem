namespace OrcaHello.Web.UI.Models
{
    public class DetectionFilterAndPagination
    {
        public string State { get; set; }
        public DateTime? FromDate { get; set; }
        public DateTime? ToDate { get; set; }
        public string SortBy { get; set; }
        public bool IsDescending { get; set; }
        public int Page { get; set; }
        public int PageSize { get; set; }
        public string Location { get; set; }
    }
}
