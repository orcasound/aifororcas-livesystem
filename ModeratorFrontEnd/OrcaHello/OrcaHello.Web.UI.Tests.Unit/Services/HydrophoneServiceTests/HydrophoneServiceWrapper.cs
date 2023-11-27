namespace OrcaHello.Web.UI.Tests.Unit.Services
{
    [ExcludeFromCodeCoverage]
    public class HydrophoneServiceWrapper : HydrophoneService
    {
        public new void ValidateResponse<T>(T response) =>
            base.ValidateResponse<T>(response);

        public new void ValidateThereAreHydrophones(int count) =>
            base.ValidateThereAreHydrophones(count);

        public new ValueTask<T> TryCatch<T>(ReturningGenericFunction<T> returningValueTaskFunction) =>
            base.TryCatch(returningValueTaskFunction);
    }
}
