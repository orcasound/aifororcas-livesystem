using Microsoft.Extensions.Logging;
using Moq;
using OrcaHello.Web.Api.Brokers.Storages;
using OrcaHello.Web.Api.Models;
using OrcaHello.Web.Api.Services;
using System.Diagnostics.CodeAnalysis;

namespace OrcaHello.Web.Api.Tests.Unit.Services
{
    [ExcludeFromCodeCoverage]
    [TestClass]
    public partial class MetadataServiceTests
    {
        private readonly Mock<IStorageBroker> _storageBrokerMock;
        private readonly Mock<ILogger<MetadataService>> _loggerMock;

        private readonly IMetadataService _metadataService;

        public MetadataServiceTests()
        {
            _storageBrokerMock = new Mock<IStorageBroker>();
            _loggerMock = new Mock<ILogger<MetadataService>>();

            _metadataService = new MetadataService(
                storageBroker: _storageBrokerMock.Object,
                logger: _loggerMock.Object);

        }

        [TestCleanup]
        public void TestTeardown()
        {
            _loggerMock.VerifyNoOtherCalls();
            _storageBrokerMock.VerifyNoOtherCalls();
        }

        public Metadata CreateRandomMetadata()
        {
            return new Metadata
            {
                Id = Guid.NewGuid().ToString(),
                State = "Unreviewed",
                LocationName = "location",
                AudioUri = "https://url",
                ImageUri = "https://url",
                Timestamp = DateTime.Now,
                WhaleFoundConfidence = 0.66M,
                Location = new Models.Location
                {
                    Name = "location",
                    Id = "location_guid",
                    Latitude = 1.00,
                    Longitude = 1.00
                },
                Predictions = new List<Prediction>
                {
                    new Prediction
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
