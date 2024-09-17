namespace OrcaHello.Web.UI.Brokers
{
    public partial interface IDetectionAPIBroker
    {
        ValueTask<DetectionListResponse> GetFilteredDetectionsAsync(string queryString);
        ValueTask<ModerateDetectionsResponse> PutModerateDetectionsAsync(ModerateDetectionsRequest request);
        ValueTask<DetectionListForTagResponse> GetFilteredDetectionsForTagAsync(string tag, string queryString);
        ValueTask<Detection> GetDetectionAsync(string id);
    }
}
