using OrcaHello.Web.Api.Services;
using OrcaHello.Web.Shared.Models.Metrics;
using System.Diagnostics.CodeAnalysis;

namespace OrcaHello.Web.Api.Tests.Unit.Services
{
    [ExcludeFromCodeCoverage]
    public class MetricsOrchestrationServiceWrapper : MetricsOrchestrationService
    {
        public new void Validate(DateTime? date, string propertyName) =>
            base.Validate(date, propertyName);

        public new ValueTask<MetricsResponse> TryCatch(ReturningMetricsResponseFunction returningMetricsResponseFunction) =>
            base.TryCatch(returningMetricsResponseFunction);
    }
}
