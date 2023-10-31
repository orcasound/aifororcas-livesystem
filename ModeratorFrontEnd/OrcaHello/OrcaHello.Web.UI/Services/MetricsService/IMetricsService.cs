namespace OrcaHello.Web.UI.Services
{
    public interface IMetricsService
    {
        ValueTask<MetricsResponse> RetrieveFilteredMetricsAsync(DateTime? fromDate, DateTime? toDate);
    }
}
