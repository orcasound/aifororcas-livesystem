namespace OrcaHello.Web.Api.Tests.Unit.Services
{
    [ExcludeFromCodeCoverage]
    [TestClass]
    public partial class DetectionOrchestrationServiceTests
    {
        private readonly Mock<IMetadataService> _metadataServiceMock;
        private readonly Mock<ILogger<DetectionOrchestrationService>> _loggerMock;

        private readonly IDetectionOrchestrationService _orchestrationService;

        public DetectionOrchestrationServiceTests()
        {
            _metadataServiceMock = new Mock<IMetadataService>();
            _loggerMock = new Mock<ILogger<DetectionOrchestrationService>>();

            _orchestrationService = new DetectionOrchestrationService(
                metadataService: _metadataServiceMock.Object,
                logger: _loggerMock.Object);
        }

        [TestCleanup]
        public void TestTeardown()
        {
            _loggerMock.VerifyNoOtherCalls();
            _metadataServiceMock.VerifyNoOtherCalls();
        }

        public Metadata CreateRandomMetadata()
        {
            return new Metadata
            {
                Id = Guid.NewGuid().ToString(),
                State = DetectionState.Unreviewed.ToString(),
                LocationName = "location",
                AudioUri = "https://url",
                ImageUri = "https://url",
                Timestamp = DateTime.Now,
                WhaleFoundConfidence = 0.66M,
                Location = new()
                {
                    Name = "location",
                    Id = "location_guid",
                    Latitude = 1.00,
                    Longitude = 1.00
                },
                Predictions = new List<Prediction>
                {
                    new()
                    {
                        Id = 1,
                        StartTime = 5.00M,
                        Duration = 1.0M,
                        Confidence = 0.66M
                    }
                },
                DateModerated = DateTime.Now.ToString()
            };
        }
    }

}
