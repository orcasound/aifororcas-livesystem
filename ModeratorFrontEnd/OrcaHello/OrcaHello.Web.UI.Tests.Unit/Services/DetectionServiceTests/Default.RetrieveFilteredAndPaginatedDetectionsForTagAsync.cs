namespace OrcaHello.Web.UI.Tests.Unit.Services
{
    public partial class DetectionServiceTests
    {
        [TestMethod]
        public async Task Default_Expect_RetrieveFilteredAndPaginatedDetectionsForTagAsync()
        {
            DetectionListForTagResponse expectedResponse = new()
            {
                Detections = new()
                {
                    new() { Id = Guid.NewGuid().ToString(), Moderator = "John Smith", Tags = new() { "Tag 1" } }
                },
                Tag = "Tag 1"
            };

            _apiBrokerMock.Setup(broker =>
                broker.GetFilteredDetectionsForTagAsync(It.IsAny<string>(), It.IsAny<string>()))
                .ReturnsAsync(expectedResponse);

            DetectionListForTagResponse actualResponse =
                await _service.RetrieveFilteredAndPaginatedDetectionsForTagAsync("Tag 1", DateTime.UtcNow.AddDays(-14), DateTime.UtcNow, 1, 10);

            Assert.AreEqual(expectedResponse, actualResponse);

            _apiBrokerMock.Verify(broker =>
                broker.GetFilteredDetectionsForTagAsync(It.IsAny<string>(), It.IsAny<string>()),
                Times.Once);
        }
    }
}