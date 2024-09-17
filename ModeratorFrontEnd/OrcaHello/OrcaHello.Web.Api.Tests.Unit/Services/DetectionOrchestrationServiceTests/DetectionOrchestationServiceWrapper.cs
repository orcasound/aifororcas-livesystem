namespace OrcaHello.Web.Api.Tests.Unit.Services
{
    [ExcludeFromCodeCoverage]
    public class DetectionOrchestrationServiceWrapper : DetectionOrchestrationService
    {
        public new void Validate(DateTime? date, string propertyName) =>
            base.Validate(date, propertyName);

        public new void Validate(string propertyValue, string propertyName) =>
            base.Validate(propertyValue, propertyName);

        public new void ValidatePage(int page) =>
            base.ValidatePage(page);

        public new void ValidatePageSize(int pageSize) =>
            base.ValidatePageSize(pageSize);

        public new void ValidateStorageMetadata(Metadata storageMetadata, string id) =>
            base.ValidateStorageMetadata(storageMetadata, id);

        public new void ValidateStateIsAcceptable(string state) =>
            base.ValidateStateIsAcceptable(state);

        public new void ValidateModerateRequestOnUpdate(ModerateDetectionsRequest request) =>
            base.ValidateModerateRequestOnUpdate(request);

        public new ValueTask<T> TryCatch<T>(ReturningGenericFunction<T> returningValueTaskFunction) =>
            base.TryCatch(returningValueTaskFunction);
    }
}
