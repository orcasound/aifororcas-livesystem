using Microsoft.Extensions.Logging;
using Moq;
using OrcaHello.Web.Api.Services;
using System.Diagnostics.CodeAnalysis;

namespace OrcaHello.Web.Api.Tests.Unit.Services
{
    [ExcludeFromCodeCoverage]
    [TestClass]
    public partial class MetricsOrchestrationServiceTests
    {
        private readonly Mock<IMetadataService> _metadataServiceMock;
        private readonly Mock<ILogger<MetricsOrchestrationService>> _loggerMock;

        private readonly IMetricsOrchestrationService _orchestrationService;

        public MetricsOrchestrationServiceTests()
        {
            _metadataServiceMock = new Mock<IMetadataService>();
            _loggerMock = new Mock<ILogger<MetricsOrchestrationService>>();

            _orchestrationService = new MetricsOrchestrationService(
                metadataService: _metadataServiceMock.Object,
                logger: _loggerMock.Object);
        }

        [TestCleanup]
        public void TestTeardown()
        {
            _loggerMock.VerifyNoOtherCalls();
            _metadataServiceMock.VerifyNoOtherCalls();
        }

    }
}