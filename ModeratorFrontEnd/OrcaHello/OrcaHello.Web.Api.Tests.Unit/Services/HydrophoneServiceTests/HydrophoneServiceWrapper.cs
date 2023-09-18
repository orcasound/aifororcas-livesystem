namespace OrcaHello.Web.Api.Tests.Unit.Services
{
    public class HydrophoneServiceWrapper : HydrophoneService
    {
        public new ValueTask<T> TryCatch<T>(ReturningGenericFunction<T> returningValueTaskFunction) =>
            base.TryCatch(returningValueTaskFunction);
    }
}
