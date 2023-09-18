namespace OrcaHello.Web.Api.Tests.Unit.Controllers
{
    public partial class DetectionsControllerTests
    {

        [TestMethod]
        public async Task Default_PutModeratedInfoAsync_Expect_Detection()
        {
            Detection expectedResult = new();

            _orchestrationServiceMock.Setup(service =>
                service.ModerateDetectionByIdAsync(It.IsAny<string>(), It.IsAny<ModerateDetectionRequest>()))
            .ReturnsAsync(expectedResult);

            ModerateDetectionRequest request = new()
            {
                Id = Guid.NewGuid().ToString(),
                Moderator = "Ira M. Goober",
                DateModerated = DateTime.UtcNow,
                Comments = "Comments",
                Tags = new List<string>() { "Tag1" }
            };

            ActionResult<Detection> actionResult =
                await _controller.PutModeratedInfoAsync(Guid.NewGuid().ToString(), request);


            var contentResult = actionResult.Result as ObjectResult;
            Assert.IsNotNull(contentResult);
            Assert.AreEqual(200, contentResult.StatusCode);

            Assert.IsNotNull((Detection)contentResult.Value);

            _orchestrationServiceMock.Verify(service =>
               service.ModerateDetectionByIdAsync(It.IsAny<string>(), It.IsAny<ModerateDetectionRequest>()),
                Times.Once);
        }
    }
}
