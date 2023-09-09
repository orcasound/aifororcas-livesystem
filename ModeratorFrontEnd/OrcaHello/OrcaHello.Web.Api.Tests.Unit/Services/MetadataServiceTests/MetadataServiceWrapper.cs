namespace OrcaHello.Web.Api.Tests.Unit.Services.Metadatas
{
    [ExcludeFromCodeCoverage]
    public class MetadataServiceWrapper : MetadataService
    {
        public new void Validate(DateTime date, string propertyName) =>
            base.Validate(date, propertyName);

        public new void Validate(string propertyValue, string propertyName) =>
            base.Validate(propertyValue, propertyName);

        public new void ValidateDatesAreWithinRange(DateTime fromDate, DateTime toDate) =>
            base.ValidateDatesAreWithinRange(fromDate, toDate);

        public new void ValidateMetadataOnCreate(Metadata metadata) =>
            base.ValidateMetadataOnCreate(metadata);

        public new void ValidateMetadataOnUpdate(Metadata metadata) =>
            base.ValidateMetadataOnUpdate(metadata);

        public new void ValidateStateIsAcceptable(string state) =>
            base.ValidateStateIsAcceptable(state);

        public new void ValidateTagContainsOnlyValidCharacters(string tag) =>
            base.ValidateTagContainsOnlyValidCharacters(tag);

        public new ValueTask<T> TryCatch<T>(ReturningGenericFunction<T> returningValueTaskFunction) =>
            base.TryCatch(returningValueTaskFunction);
    }
}
