namespace OrcaHello.Web.Api.Tests.Unit.Controllers
{
    public partial class InterestLabelsControllerTests
    {
        [TestMethod]
        public async Task Default_AddInterestLabelToDetectionAsync_Expect_Detection()
        {
            InterestLabelAddResponse expectedResult = new();

            _orchestrationServiceMock.Setup(service =>
                service.AddInterestLabelToDetectionAsync(It.IsAny<string>(), It.IsAny<string>()))
            .ReturnsAsync(expectedResult);

            ActionResult<InterestLabelAddResponse> actionResult =
                await _controller.AddInterestLabelToDetectionAsync("id", "label");


            var contentResult = actionResult.Result as ObjectResult;
            Assert.IsNotNull(contentResult);
            Assert.AreEqual(200, contentResult.StatusCode);

            Assert.IsNotNull((InterestLabelAddResponse)contentResult.Value!);

            _orchestrationServiceMock.Verify(service =>
               service.AddInterestLabelToDetectionAsync(It.IsAny<string>(), It.IsAny<string>()),
                Times.Once);
        }
    }
}
