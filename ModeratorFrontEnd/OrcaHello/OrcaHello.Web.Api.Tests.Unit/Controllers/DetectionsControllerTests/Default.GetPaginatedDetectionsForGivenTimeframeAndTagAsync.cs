namespace OrcaHello.Web.Api.Tests.Unit.Controllers
{
    public partial class DetectionsControllerTests
    {
        [TestMethod]
        public async Task Default_GetPaginatedDetectionsForGivenTimeframeAndTagAsync_Expect_DetectionListResponse()
        {
            DetectionListForTagResponse response = new()
            {
                Count = 2,
                FromDate = DateTime.Now,
                ToDate = DateTime.Now.AddDays(1),
                Detections = new List<Detection> { new() }
            };

            _orchestrationServiceMock.Setup(service =>
                service.RetrieveDetectionsForGivenTimeframeAndTagAsync(It.IsAny<DateTime>(), It.IsAny<DateTime>(), It.IsAny<string>(), It.IsAny<int>(), It.IsAny<int>()))
                .ReturnsAsync(response);

            ActionResult<DetectionListForTagResponse> actionResult =
                await _controller.GetPaginatedDetectionsForGivenTimeframeAndTagAsync("tag", DateTime.Now, DateTime.Now.AddDays(1), 1, 10);

            var contentResult = actionResult.Result as ObjectResult;
            Assert.IsNotNull(contentResult);
            Assert.AreEqual(200, contentResult.StatusCode);

            Assert.AreEqual(response.Detections.Count,
                ((DetectionListForTagResponse)contentResult.Value!).Detections.Count);

            _orchestrationServiceMock.Verify(service =>
               service.RetrieveDetectionsForGivenTimeframeAndTagAsync(It.IsAny<DateTime>(), It.IsAny<DateTime>(), It.IsAny<string>(), It.IsAny<int>(), It.IsAny<int>()),
                Times.Once);
        }
    }
}
