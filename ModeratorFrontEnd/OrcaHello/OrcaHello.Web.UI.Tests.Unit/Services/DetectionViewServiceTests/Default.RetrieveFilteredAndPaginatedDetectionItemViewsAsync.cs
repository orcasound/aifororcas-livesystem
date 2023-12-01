using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace OrcaHello.Web.UI.Tests.Unit.Services
{
    public partial class DetectionViewServiceTests
    {
        [TestMethod]
        public async Task Default_Expect_RetrieveFilteredAndPaginatedDetectionItemViewsAsync()
        {
            DetectionListResponse expectedResponse = new()
            {
                Detections = new()
                {
                    new() { Id = Guid.NewGuid().ToString(), Moderator = "John Smith", Tags = new() { "Tag 1" }, Location = new() { Name = "Test" }  }
                }
            };

            _detectionServiceMock.Setup(service =>
                service.RetrieveFilteredAndPaginatedDetectionsAsync(
                    It.IsAny<string>(), It.IsAny<DateTime>(), It.IsAny<DateTime>(), It.IsAny<string>(), It.IsAny<bool>(), It.IsAny<int>(), It.IsAny<int>(), It.IsAny<string>()))
                    .ReturnsAsync(expectedResponse);

            PaginatedDetectionsByStateRequest request = new()
            {
                State = "Positive",
                FromDate = DateTime.UtcNow.AddDays(-14),
                ToDate = DateTime.UtcNow,
                Page = 1,
                PageSize = 10,
                Location = "Location 1",
                SortBy = "Timestamp",
                IsDescending = true
            };

            DetectionItemViewResponse actualResponse =
                await _viewService.RetrieveFilteredAndPaginatedDetectionItemViewsAsync(request);

            Assert.AreEqual(expectedResponse.Detections.Count(), actualResponse.DetectionItemViews.Count());

            _detectionServiceMock.Verify(service =>
                service.RetrieveFilteredAndPaginatedDetectionsAsync(
                    It.IsAny<string>(), It.IsAny<DateTime>(), It.IsAny<DateTime>(), It.IsAny<string>(), It.IsAny<bool>(), It.IsAny<int>(), It.IsAny<int>(), It.IsAny<string>()),
                    Times.Once);
        }
    }
}