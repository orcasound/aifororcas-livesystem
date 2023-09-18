namespace OrcaHello.Web.Api.Services
{
    public interface IInterestLabelOrchestrationService
    {
        ValueTask<InterestLabelListResponse> RetrieveAllInterestLabelsAsync();
        ValueTask<InterestLabelRemovalResponse> RemoveInterestLabelFromDetectionAsync(string id);
        ValueTask<InterestLabelAddResponse> AddInterestLabelToDetectionAsync(string id, string interestLabel);
    }
}
