namespace OrcaHello.Web.UI.Tests.Unit.Services
{
    public class DashboardViewServiceWrapper : DashboardViewService
    {
        public new void ValidateDateRange(DateTime? fromDate, DateTime? toDate) =>
            base.ValidateDateRange(fromDate, toDate);

        public new void ValidatePagination(int page, int pageSize) =>
            base.ValidatePagination(page, pageSize);

        public new void ValidateModerator(string moderator) =>
            base.ValidateModerator(moderator);

        public new void ValidateTag(string tag) =>
            base.ValidateTag(tag);

        public new void ValidateResponse<T>(T response) =>
            base.ValidateResponse(response);

        public new void ValidateRequest<T>(T request) =>
            base.ValidateRequest(request);

        public new ValueTask<T> TryCatch<T>(ReturningGenericFunction<T> returningValueTaskFunction) =>
            base.TryCatch(returningValueTaskFunction);
    }
}
