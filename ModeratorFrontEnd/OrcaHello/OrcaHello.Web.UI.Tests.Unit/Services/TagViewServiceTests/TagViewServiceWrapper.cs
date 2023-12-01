namespace OrcaHello.Web.UI.Tests.Unit.Services
{
    public class TagViewServiceWrapper : TagViewService
    {
        public new void ValidateAtLeastOneTagSelected(List<string> tags) =>
            base.ValidateAtLeastOneTagSelected(tags);

        public new void ValidateDateRange(DateTime? fromDate, DateTime? toDate) =>
            base.ValidateDateRange(fromDate, toDate);

        public new void ValidatePagination(int page, int pageSize) =>
            base.ValidatePagination(page, pageSize);

        public new void ValidateTagString(string tag, string name) =>
            base.ValidateTagString(tag, name);

        public new void ValidateRequest<T>(T request) =>
            base.ValidateRequest(request);

        public new void ValidateResponse<T>(T response) =>
            base.ValidateResponse(response);

        public new ValueTask<T> TryCatch<T>(ReturningGenericFunction<T> returningValueTaskFunction) =>
            base.TryCatch(returningValueTaskFunction);
    }
}
