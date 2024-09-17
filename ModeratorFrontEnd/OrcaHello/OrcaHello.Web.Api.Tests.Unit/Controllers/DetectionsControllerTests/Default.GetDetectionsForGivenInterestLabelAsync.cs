namespace OrcaHello.Web.Api.Tests.Unit.Controllers
{
    public partial class DetectionsControllerTests
    {
        [TestMethod]
        public async Task Default_GetDetectionsForGivenInterestLabelAsyncc_Expect_DetectionListForInterestLabelResponse()
        {
            DetectionListForInterestLabelResponse response = new()
            {
                TotalCount = 1,
                Detections = new List<Detection> { new() },
                InterestLabel = "test"
            };

            _orchestrationServiceMock.Setup(service =>
                service.RetrieveDetectionsForGivenInterestLabelAsync(It.IsAny<string>()))
                .ReturnsAsync(response);

            ActionResult<DetectionListForInterestLabelResponse> actionResult =
                await _controller.GetDetectionsForGivenInterestLabelAsync("test");

            var contentResult = actionResult.Result as ObjectResult;
            Assert.IsNotNull(contentResult);
            Assert.AreEqual(200, contentResult.StatusCode);

            Assert.AreEqual(response.Detections.Count,
                ((DetectionListForInterestLabelResponse)contentResult.Value!).Detections.Count);

            _orchestrationServiceMock.Verify(service =>
               service.RetrieveDetectionsForGivenInterestLabelAsync(It.IsAny<string>()),
                Times.Once);
        }
    }
}
