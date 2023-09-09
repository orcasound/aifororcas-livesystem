namespace OrcaHello.Web.Api.Tests.Unit.Services
{
    [ExcludeFromCodeCoverage]
    public class TagOrchestrationServiceWrapper : TagOrchestrationService
    {
        public new void Validate(DateTime? date, string propertyName) =>
            base.Validate(date, propertyName);

        public new void Validate(string propertyValue, string propertyName) =>
            base.Validate(propertyValue, propertyName);

        public new ValueTask<T> TryCatch<T>(ReturningGenericFunction<T> returningValueTaskFunction) =>
            base.TryCatch(returningValueTaskFunction);
    }
}
