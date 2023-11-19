namespace OrcaHello.Web.Api.Models
{
    [ExcludeFromCodeCoverage]
    public class QueryableMetadata
    {
        public IQueryable<Metadata> QueryableRecords { get; set; }
        public int TotalCount { get; set; }
    }

    [ExcludeFromCodeCoverage]
    public class QueryableMetadataForTimeframeAndTag : QueryableMetadata
    {
        public DateTime FromDate { get; set; }
        public DateTime ToDate { get; set; }
        public string Tag { get; set; }
        public int Page { get; set; }
        public int PageSize { get; set; }
    }

    [ExcludeFromCodeCoverage]
    public class QueryableMetadataForTimeframe : QueryableMetadata
    {
        public DateTime FromDate { get; set; }
        public DateTime ToDate { get; set; }
        public int Page { get; set; }
        public int PageSize { get; set; }
    }

    [ExcludeFromCodeCoverage]
    public class QueryableMetadataForTimeframeAndModerator : QueryableMetadata
    {
        public DateTime FromDate { get; set; }
        public DateTime ToDate { get; set; }
        public int Page { get; set; }
        public int PageSize { get; set; }
        public string Moderator { get; set; }
    }

    [ExcludeFromCodeCoverage]
    public class QueryableMetadataForTimeframeTagAndModerator : QueryableMetadata
    {
        public DateTime FromDate { get; set; }
        public DateTime ToDate { get; set; }
        public int Page { get; set; }
        public int PageSize { get; set; }
        public string Moderator { get; set; }
        public string Tag { get; set; }
    }

    [ExcludeFromCodeCoverage]
    public class QueryableMetadataFiltered : QueryableMetadata
    {
        public DateTime FromDate { get; set; }
        public DateTime ToDate { get; set; }
        public int Page { get; set; }
        public int PageSize { get; set; }
        public string State { get; set; }
        public string SortBy { get; set; }
        public string SortOrder { get; set; }
        public string Location { get; set; }
    }
}
