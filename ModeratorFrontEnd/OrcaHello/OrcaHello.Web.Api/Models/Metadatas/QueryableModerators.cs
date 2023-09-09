namespace OrcaHello.Web.Api.Models
{
    [ExcludeFromCodeCoverage]
    public class QueryableModerators
    {
        public IQueryable<string> QueryableRecords { get; set; }
        public int TotalCount { get; set; }
    }
}
