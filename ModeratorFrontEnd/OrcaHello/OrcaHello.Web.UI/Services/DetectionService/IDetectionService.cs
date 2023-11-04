namespace OrcaHello.Web.UI.Services
{
    public interface IDetectionService
    {
        ValueTask<DetectionListResponse> RetrieveFilteredAndPaginatedDetectionsAsync(string state, DateTime? fromDate, DateTime? toDate,
            string sortBy, bool isDescending, int page, int pageSize, string location);

        ValueTask<DetectionListForTagResponse> RetrieveFilteredAndPaginatedDetectionsForTagAsync(string tag, DateTime? fromDate, DateTime? toDate,
            int page, int pageSize);

        ValueTask<ModerateDetectionsResponse> ModerateDetectionsAsync(ModerateDetectionsRequest request);

        ValueTask<Detection> RetrieveDetectionAsync(string id);
    }
}
