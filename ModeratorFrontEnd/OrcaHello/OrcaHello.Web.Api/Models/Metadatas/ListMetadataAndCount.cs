namespace OrcaHello.Web.Api.Models
{
    [ExcludeFromCodeCoverage]
    public class ListMetadataAndCount
    {
        public List<Metadata> PaginatedRecords { get; set; } = new List<Metadata>();
        public int TotalCount { get; set; }
    }
}
