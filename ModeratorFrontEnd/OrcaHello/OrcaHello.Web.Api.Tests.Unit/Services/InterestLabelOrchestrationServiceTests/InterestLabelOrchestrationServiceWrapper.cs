namespace OrcaHello.Web.Api.Tests.Unit.Services
{
    [ExcludeFromCodeCoverage]
    public class InterestLabelOrchestrationServiceWrapper : InterestLabelOrchestrationService
    {
        public new void Validate(string propertyValue, string propertyName) =>
            base.Validate(propertyValue, propertyName);

        public new void ValidateMetadataFound(Metadata metadata, string id) =>
            base.ValidateMetadataFound(metadata, id);

        public new void ValidateDeleted(bool deleted, string id) =>
            base.ValidateDeleted(deleted, id);

        public new void ValidateInserted(bool inserted, string id) =>
            base.ValidateInserted(inserted, id);

        public new ValueTask<T> TryCatch<T>(ReturningGenericFunction<T> returningValueTaskFunction) =>
            base.TryCatch(returningValueTaskFunction);
    }
}
