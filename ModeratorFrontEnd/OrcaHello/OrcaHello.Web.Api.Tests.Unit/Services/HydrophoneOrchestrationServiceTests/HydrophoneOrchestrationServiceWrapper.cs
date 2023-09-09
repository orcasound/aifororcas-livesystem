using OrcaHello.Web.Api.Services;
using System.Diagnostics.CodeAnalysis;

namespace OrcaHello.Web.Api.Tests.Unit.Services
{
    [ExcludeFromCodeCoverage]
    public class HydrophoneOrchestrationServiceWrapper : HydrophoneOrchestrationService
    {
        public new T TryCatch<T>(ReturningGenericFunction<T> returningValueTaskFunction) =>
            base.TryCatch(returningValueTaskFunction);
    }
}
