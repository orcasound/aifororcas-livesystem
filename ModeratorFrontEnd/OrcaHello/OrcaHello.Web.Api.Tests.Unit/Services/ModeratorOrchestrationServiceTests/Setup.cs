using Microsoft.Extensions.Logging;
using Moq;
using OrcaHello.Web.Api.Models;
using OrcaHello.Web.Api.Services;
using System.Diagnostics.CodeAnalysis;

namespace OrcaHello.Web.Api.Tests.Unit.Services
{
    [ExcludeFromCodeCoverage]
    [TestClass]
    public partial class ModeratorOrchestrationServiceTests
    {
        private readonly Mock<IMetadataService> _metadataServiceMock;
        private readonly Mock<ILogger<ModeratorOrchestrationService>> _loggerMock;

        private readonly IModeratorOrchestrationService _orchestrationService;

        public ModeratorOrchestrationServiceTests()
        {
            _metadataServiceMock = new Mock<IMetadataService>();
            _loggerMock = new Mock<ILogger<ModeratorOrchestrationService>>();

            _orchestrationService = new ModeratorOrchestrationService(
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
                State = "Positive",
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
                DateModerated = DateTime.Now.ToString(),
                Moderator = "Moderator",
                Comments = "Comments are here"
            };
        }

    }
}
