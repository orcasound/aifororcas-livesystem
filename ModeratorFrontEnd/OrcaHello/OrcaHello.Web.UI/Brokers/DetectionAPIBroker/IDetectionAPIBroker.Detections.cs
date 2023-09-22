namespace OrcaHello.Web.UI.Brokers
{
    public partial interface IDetectionAPIBroker
    {
        ValueTask<DetectionListResponse> GetFilteredDetectionsAsync(string queryString);
    }
}
