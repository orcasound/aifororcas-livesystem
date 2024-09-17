namespace OrcaHello.Web.UI.Models
{
    [ExcludeFromCodeCoverage]
    public class MetricsStateView
    {
        public DateTime FromDate { get; set; }
        public DateTime ToDate { get; set; }
        public bool IsLoading { get; set; } = false;
        public List<MetricsItemView> MetricsItemViews { get; set; } = null!;
        public List<string> FillColors { get; set; } = new();

        public void Reset(DateTime fromDate, DateTime toDate)
        {
            IsLoading = false;
            FromDate = fromDate;
            ToDate = toDate;
        }

        public bool IsPopulated => MetricsItemViews != null && MetricsItemViews.Sum(p => p.Value) > 0;

        public bool IsEmpty => MetricsItemViews != null && MetricsItemViews.Sum(p => p.Value) == 0;
    }
}
