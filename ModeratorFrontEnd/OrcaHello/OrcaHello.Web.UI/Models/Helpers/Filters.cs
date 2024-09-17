namespace OrcaHello.Web.UI.Models
{
    [ExcludeFromCodeCoverage]
    public class Filters
    {
        public DetectionState DetectionState { get; set; }
        public SortOrder SortOrder { get; set; }
        public SortBy SortBy { get; set; }
        public string Location { get; set; } = string.Empty;
        public int MaxRecords { get; set; } = 0;
        public Timeframe Timeframe { get; set; }
        public DateTime? StartDateTime { get; set; }
        public DateTime? EndDateTime { get; set; }
    }
}
