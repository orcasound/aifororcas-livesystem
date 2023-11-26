namespace OrcaHello.Web.UI.Models
{
    [ExcludeFromCodeCoverage]
    public class MetricsItemViewResponse
    {
        public List<MetricsItemView> MetricsItemViews { get; set; } = new();
        public DateTime FromDate { get; set; }
        public DateTime ToDate { get; set; }
    }
}
