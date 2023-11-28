namespace OrcaHello.Web.UI.Tests.Unit.Services
{
    [ExcludeFromCodeCoverage]
    public class MetricsServiceWrapper : MetricsService
    {
        public new void ValidateDateRange(DateTime? fromDate, DateTime? toDate) =>
            base.ValidateDateRange(fromDate, toDate);

        public new void ValidateResponse<T>(T response) =>
            base.ValidateResponse<T>(response);

        public new ValueTask<T> TryCatch<T>(ReturningGenericFunction<T> returningValueTaskFunction) =>
            base.TryCatch(returningValueTaskFunction);
    }
}
