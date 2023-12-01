namespace OrcaHello.Web.UI.Tests.Unit.Services
{
    public class HydrophoneViewServiceWrapper : HydrophoneViewService
    {
        public new void ValidateResponse<T>(T response) =>
            base.ValidateResponse(response);

        public new ValueTask<T> TryCatch<T>(ReturningGenericFunction<T> returningValueTaskFunction) =>
            base.TryCatch(returningValueTaskFunction);
    }
}
