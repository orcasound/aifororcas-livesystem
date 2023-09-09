namespace OrcaHello.Web.Api.Services
{
    public interface IMetricsOrchestrationService
    {
        ValueTask<MetricsResponse> RetrieveMetricsForGivenTimeframeAsync(DateTime? fromDate, DateTime? toDate);
    }
}
