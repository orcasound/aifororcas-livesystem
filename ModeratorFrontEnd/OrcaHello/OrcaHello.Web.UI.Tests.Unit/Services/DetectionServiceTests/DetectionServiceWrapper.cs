namespace OrcaHello.Web.UI.Tests.Unit.Services
{
    public class DetectionServiceWrapper : DetectionService
    {
        public new void Validate(string propertyValue, string propertyName) =>
            base.Validate(propertyValue, propertyName);

        public new void Validate(DateTime? date, string propertyName) =>
            base.Validate(date, propertyName);

        public new void ValidateAtLeastOneId(List<string> items, string propertyName) =>
            base.ValidateAtLeastOneId(items, propertyName);

        public new void ValidateDateRange(DateTime? fromDate, DateTime? toDate) =>
            base.ValidateDateRange(fromDate, toDate);

        public new void ValidatePagination(int page, int pageSize) =>
            base.ValidatePagination(page, pageSize);

        public new void ValidateRequest<T>(T response) =>
            base.ValidateRequest<T>(response);

        public new void ValidateResponse<T>(T response) =>
            base.ValidateResponse<T>(response);

        public new ValueTask<T> TryCatch<T>(ReturningGenericFunction<T> returningValueTaskFunction) =>
            base.TryCatch(returningValueTaskFunction);
    }
}
