namespace OrcaHello.Web.Api.Tests.Unit.Controllers
{
    [TestClass]
    [ExcludeFromCodeCoverage]
    public partial class DetectionsControllerTests
    {
        private readonly Mock<IDetectionOrchestrationService> _orchestrationServiceMock;
        private readonly DetectionsController _controller;

        public DetectionsControllerTests()
        {
            _orchestrationServiceMock = new Mock<IDetectionOrchestrationService>();

            _controller = new DetectionsController(
                detectionOrchestrationService: _orchestrationServiceMock.Object);
        }

        [TestCleanup]
        public void TestTeardown()
        {
            _orchestrationServiceMock.VerifyNoOtherCalls();
        }
    }
}
