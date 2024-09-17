namespace OrcaHello.Web.Api.Services
{
    public interface IDetectionOrchestrationService
    {
        ValueTask<DetectionListForTagResponse> RetrieveDetectionsForGivenTimeframeAndTagAsync(DateTime? fromDate, DateTime? toDate, string tag, int page, int pageSize);
        ValueTask<DetectionListResponse> RetrieveFilteredDetectionsAsync(DateTime? fromDate, DateTime? toDate, string state, string sortBy, bool isDescending, string location, int page, int pageSize);
        ValueTask<Detection> RetrieveDetectionByIdAsync(string id);
        ValueTask<ModerateDetectionsResponse> ModerateDetectionsByIdAsync(ModerateDetectionsRequest request);
        ValueTask<DetectionListForInterestLabelResponse> RetrieveDetectionsForGivenInterestLabelAsync(string interestLabel);
    }
}
