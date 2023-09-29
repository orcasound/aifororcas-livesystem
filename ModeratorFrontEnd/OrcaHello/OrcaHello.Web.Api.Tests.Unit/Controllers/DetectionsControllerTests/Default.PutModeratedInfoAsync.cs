namespace OrcaHello.Web.Api.Tests.Unit.Controllers
{
    public partial class DetectionsControllerTests
    {

        [TestMethod]
        public async Task Default_PutModeratedInfoAsync_Expect_Detection()
        {
            ModerateDetectionsResponse expectedResult = new();

            _orchestrationServiceMock.Setup(service =>
                service.ModerateDetectionsByIdAsync(It.IsAny<ModerateDetectionsRequest>()))
            .ReturnsAsync(expectedResult);

            ModerateDetectionsRequest request = new()
            {
                Ids = new List<string> { Guid.NewGuid().ToString() },
                Moderator = "Ira M. Goober",
                DateModerated = DateTime.UtcNow,
                Comments = "Comments",
                Tags = new List<string>() { "Tag1" }
            };

            ActionResult<ModerateDetectionsResponse> actionResult =
                await _controller.PutModeratedInfoAsync(request);


            var contentResult = actionResult.Result as ObjectResult;
            Assert.IsNotNull(contentResult);
            Assert.AreEqual(200, contentResult.StatusCode);

            if (contentResult.Value is ModerateDetectionsResponse response)
            {
                Assert.IsNotNull(response);
            }

            _orchestrationServiceMock.Verify(service =>
               service.ModerateDetectionsByIdAsync(It.IsAny<ModerateDetectionsRequest>()),
                Times.Once);
        }
    }
}
