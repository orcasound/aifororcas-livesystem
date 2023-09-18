namespace OrcaHello.Web.Api.Models
{
    [ExcludeFromCodeCoverage]
    public class MetricsSummaryForTimeframe
    {
        public IQueryable<MetricResult> QueryableRecords { get; set; }
        public DateTime FromDate { get; set; }
        public DateTime ToDate { get; set; }
    }

    [ExcludeFromCodeCoverage]
    public class MetricsSummaryForTimeframeAndModerator : MetricsSummaryForTimeframe
    {
        public string Moderator { get; set; }
    }
}
