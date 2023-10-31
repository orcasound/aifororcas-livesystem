namespace OrcaHello.Web.UI.Brokers
{
    public partial interface IDetectionAPIBroker
    {
        ValueTask<MetricsResponse> GetFilteredMetricsAsync(string queryString);
    }
}
