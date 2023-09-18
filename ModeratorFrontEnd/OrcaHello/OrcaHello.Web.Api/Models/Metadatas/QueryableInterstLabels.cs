namespace OrcaHello.Web.Api.Models
{
    [ExcludeFromCodeCoverage]
    public class QueryableInterestLabels
    {
        public IQueryable<string> QueryableRecords { get; set; }
        public int TotalCount { get; set; }
    }
}
