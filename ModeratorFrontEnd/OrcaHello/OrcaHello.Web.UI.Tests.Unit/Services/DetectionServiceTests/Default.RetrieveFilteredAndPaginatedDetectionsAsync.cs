using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace OrcaHello.Web.UI.Tests.Unit.Services
{
    public partial class DetectionServiceTests
    {
        [TestMethod]
        public async Task Default_Expect_RetrieveFilteredAndPaginatedDetectionsAsync()
        {
            DetectionListResponse expectedResponse = new()
            {
                Detections = new()
                {
                    new() { Id = Guid.NewGuid().ToString(), Moderator = "John Smith", Tags = new() { "Tag 1" } }
                }
            };

            _apiBrokerMock.Setup(broker =>
                broker.GetFilteredDetectionsAsync(It.IsAny<string>()))
                .ReturnsAsync(expectedResponse);

            DetectionListResponse actualResponse =
                await _service.RetrieveFilteredAndPaginatedDetectionsAsync("Positive", DateTime.UtcNow.AddDays(-14), DateTime.UtcNow,
                    "Timestamp", true, 1, 10, "All");

            Assert.AreEqual(expectedResponse, actualResponse);

            _apiBrokerMock.Verify(broker =>
                broker.GetFilteredDetectionsAsync(It.IsAny<string>()),
                Times.Once);
        }

        [TestMethod]
        public async Task Default_Expect_RetrieveFilteredAndPaginatedDetectionsAsync_With_Null_Location()
        {
            DetectionListResponse expectedResponse = new()
            {
                Detections = new()
                {
                    new() { Id = Guid.NewGuid().ToString(), Moderator = "John Smith", Tags = new() { "Tag 1" } }
                }
            };

            _apiBrokerMock.Setup(broker =>
                broker.GetFilteredDetectionsAsync(It.IsAny<string>()))
                .ReturnsAsync(expectedResponse);

            DetectionListResponse actualResponse =
                await _service.RetrieveFilteredAndPaginatedDetectionsAsync("Positive", DateTime.UtcNow.AddDays(-14), DateTime.UtcNow,
                    "Timestamp", true, 1, 10, string.Empty);

            Assert.AreEqual(expectedResponse, actualResponse);

            _apiBrokerMock.Verify(broker =>
                broker.GetFilteredDetectionsAsync(It.IsAny<string>()),
                Times.Once);
        }
    }
}