namespace OrcaHello.Web.Api.Tests.Unit.Controllers
{
    [TestClass]
    [ExcludeFromCodeCoverage]
    public partial class MetricsControllerTests
    {
        private readonly Mock<IMetricsOrchestrationService> _orchestrationServiceMock;
        private readonly MetricsController _controller;

        public MetricsControllerTests()
        {
            _orchestrationServiceMock = new Mock<IMetricsOrchestrationService>();

            _controller = new MetricsController(
                metricsOrchestrationService: _orchestrationServiceMock.Object);
        }

        [TestCleanup]
        public void TestTeardown()
        {
            _orchestrationServiceMock.VerifyNoOtherCalls();
        }
    }
}