namespace OrcaHello.Web.UI.Tests.Unit.Services
{
    [ExcludeFromCodeCoverage]
    public class TagServiceWrapper : TagService
    {
        public new void Validate(string propertyValue, string propertyName) =>
            base.Validate(propertyValue, propertyName);

        public new void ValidateDateRange(DateTime? fromDate, DateTime? toDate) =>
            base.ValidateDateRange(fromDate, toDate);

        public new void ValidateRequest<T>(T response) =>
            base.ValidateRequest(response);

        public new void ValidateResponse<T>(T response) =>
            base.ValidateResponse<T>(response);

        public new ValueTask<T> TryCatch<T>(ReturningGenericFunction<T> returningValueTaskFunction) =>
            base.TryCatch(returningValueTaskFunction);
    }
}