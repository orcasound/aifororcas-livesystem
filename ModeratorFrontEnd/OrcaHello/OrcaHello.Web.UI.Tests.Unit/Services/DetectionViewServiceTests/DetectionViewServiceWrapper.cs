namespace OrcaHello.Web.UI.Tests.Unit.Services
{
    public class DetectionViewServiceWrapper : DetectionViewService
    {
        public new void Validate(string property, string propertyName) =>
            base.Validate(property, propertyName);

        public new void ValidateAtLeastOneId(List<string> ids) =>
            base.ValidateAtLeastOneId(ids);

        public new void ValidateDateRange(DateTime? fromDate, DateTime? toDate) =>
            base.ValidateDateRange(fromDate, toDate);

        public new void ValidatePagination(int page, int pageSize) =>
            base.ValidatePagination(page, pageSize);

        public new void ValidateRequest<T>(T request) =>
            base.ValidateRequest(request);

        public new void ValidateResponse<T>(T response) =>
            base.ValidateResponse(response);

        public new ValueTask<T> TryCatch<T>(ReturningGenericFunction<T> returningValueTaskFunction) =>
            base.TryCatch(returningValueTaskFunction);
    }
}
