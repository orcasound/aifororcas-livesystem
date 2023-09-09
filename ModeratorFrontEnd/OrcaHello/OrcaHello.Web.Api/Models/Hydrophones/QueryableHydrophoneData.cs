namespace OrcaHello.Web.Api.Models
{
    [ExcludeFromCodeCoverage]
    public class QueryableHydrophoneData
    {
        public IQueryable<HydrophoneData> QueryableRecords { get; set; }
        public int TotalCount { get; set; }
    }
}
