namespace OrcaHello.Web.UI.Tests.Unit.Services
{
    public partial class DashboardViewServiceTests
    {
        [TestMethod]
        public async Task Default_Expect_RetrieveFilteredDetectionsForTagsAsync()
        {
            DetectionListForTagResponse expectedResponse = new()
            {
                Detections = new()
                {
                    new() { Id = Guid.NewGuid().ToString(), Moderator = "John Smith", Tags = new() { "Tag 1" }, Location = new() { Name = "Test" }  }
                },
                Tag = "Tag1",
                Count = 1,
                TotalCount = 1
            };

            _detectionServiceMock.Setup(service =>
                service.RetrieveFilteredAndPaginatedDetectionsForTagAsync(
                    It.IsAny<string>(), It.IsAny<DateTime>(), It.IsAny<DateTime>(), It.IsAny<int>(), It.IsAny<int>()))
                    .ReturnsAsync(expectedResponse);

            PaginatedDetectionsByTagAndDateRequest request = new()
            {
                Tag = "Tag1",
                FromDate = DateTime.UtcNow.AddDays(-14),
                ToDate = DateTime.UtcNow,
                Page = 1,
                PageSize = 10,
            };

            DetectionItemViewResponse actualResponse =
                await _viewService.RetrieveFilteredDetectionsForTagsAsync(request);

            Assert.AreEqual(expectedResponse.TotalCount, actualResponse.Count);

            _detectionServiceMock.Verify(service =>
                service.RetrieveFilteredAndPaginatedDetectionsForTagAsync(
                    It.IsAny<string>(), It.IsAny<DateTime>(), It.IsAny<DateTime>(), It.IsAny<int>(), It.IsAny<int>()),
                    Times.Once);
        }
    }
}