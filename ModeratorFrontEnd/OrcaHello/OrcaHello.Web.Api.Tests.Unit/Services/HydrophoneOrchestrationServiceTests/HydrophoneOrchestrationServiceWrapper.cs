namespace OrcaHello.Web.Api.Tests.Unit.Services
{
    [ExcludeFromCodeCoverage]
    public class HydrophoneOrchestrationServiceWrapper : HydrophoneOrchestrationService
    {
        public new ValueTask<T> TryCatch<T>(ReturningGenericFunction<T> returningValueTaskFunction) =>
            base.TryCatch(returningValueTaskFunction);
    }
}
