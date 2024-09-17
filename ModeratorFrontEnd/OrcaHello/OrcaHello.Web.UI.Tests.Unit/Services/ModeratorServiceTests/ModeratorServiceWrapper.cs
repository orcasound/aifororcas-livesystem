namespace OrcaHello.Web.UI.Tests.Unit.Services
{
    public class ModeratorServiceWrapper : ModeratorService
    {
        public new void Validate(string propertyValue, string propertyName) =>
            base.Validate(propertyValue, propertyName);

        public new void ValidateDateRange(DateTime? fromDate, DateTime? toDate) =>
            base.ValidateDateRange(fromDate, toDate);

        public new void ValidatePagination(int page, int pageSize) =>
            base.ValidatePagination(page, pageSize);

        public new void ValidateResponse<T>(T response) =>
            base.ValidateResponse<T>(response);

        public new ValueTask<T> TryCatch<T>(ReturningGenericFunction<T> returningValueTaskFunction) =>
            base.TryCatch(returningValueTaskFunction);
    }
}
