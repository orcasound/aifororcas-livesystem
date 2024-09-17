namespace OrcaHello.Web.UI.Tests.Unit.Services
{
    public partial class DetectionViewServiceTests
    {
        [TestMethod]
        public async Task Default_Expect_RetrieveDetectionAsync()
        {
            var id = Guid.NewGuid().ToString();

            Detection expectedResponse = new()
            {
                Id = id,
                Comments = "Comments",
                Annotations = new()
                {
                    new() { Id = 1, Confidence = 95, StartTime = 10, EndTime = 5}
                },
                Location = new() { Name = "Test" }
            };

            _detectionServiceMock.Setup(service =>
                service.RetrieveDetectionAsync(It.IsAny<string>()))
                .ReturnsAsync(expectedResponse);

            DetectionItemView actualResponse =
                await _viewService.RetrieveDetectionAsync(id);

            Assert.AreEqual(expectedResponse.Id, actualResponse.Id);

            _detectionServiceMock.Verify(service =>
                service.RetrieveDetectionAsync(It.IsAny<string>()),
                Times.Once);
        }
    }
}