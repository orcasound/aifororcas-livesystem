namespace OrcaHello.Web.Api.Models
{
    [ExcludeFromCodeCoverage]
    public class QueryableTags
    {
        public IQueryable<string> QueryableRecords { get; set; }
        public int TotalCount { get; set; }
    }

    [ExcludeFromCodeCoverage]
    public class QueryableTagsForTimeframe : QueryableTags
    {
        public DateTime FromDate { get; set; }
        public DateTime ToDate { get; set; }
    }

    [ExcludeFromCodeCoverage]
    public class QueryableTagsForTimeframeAndModerator : QueryableTags
    {
        public DateTime FromDate { get; set; }
        public DateTime ToDate { get; set; }
        public string Moderator { get; set; }
    }
}
