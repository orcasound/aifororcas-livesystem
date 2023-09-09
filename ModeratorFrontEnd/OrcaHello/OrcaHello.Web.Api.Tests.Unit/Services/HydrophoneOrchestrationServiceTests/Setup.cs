using Microsoft.Extensions.Logging;
using Moq;
using OrcaHello.Web.Api.Models.Configurations;
using OrcaHello.Web.Api.Services;
using System.Diagnostics.CodeAnalysis;

namespace OrcaHello.Web.Api.Tests.Unit.Services
{
    [ExcludeFromCodeCoverage]
    [TestClass]
    public partial class HydrophoneOrchestrationServiceTests
    {
        private readonly Mock<ILogger<HydrophoneOrchestrationService>> _loggerMock;
        private readonly Mock<AppSettings> _appSettingsMock;

        private readonly IHydrophoneOrchestrationService _orchestrationService;

        public HydrophoneOrchestrationServiceTests()
        {
            _loggerMock = new Mock<ILogger<HydrophoneOrchestrationService>>();
            _appSettingsMock = new Mock<AppSettings>();

            _orchestrationService = new HydrophoneOrchestrationService(
                appSettings: _appSettingsMock.Object,
                logger: _loggerMock.Object);
        }

        [TestCleanup]
        public void TestTeardown()
        {
            _loggerMock.VerifyNoOtherCalls();
            _appSettingsMock.VerifyNoOtherCalls();
        }
    }
}

