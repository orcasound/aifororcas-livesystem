namespace OrcaHello.Web.Api.Tests.Unit.Controllers
{
    public partial class InterestLabelsControllerTests
    {
        [TestMethod]
        public async Task Default_RemoveInterestLabelFromDetectionAsync_Expect_TagRemovalResponse()
        {
            var labelToRemove = "labelToRemove";

            InterestLabelRemovalResponse response = new()
            {
                Id = "id",
                LabelRemoved = labelToRemove
            };

            _orchestrationServiceMock.Setup(service =>
                service.RemoveInterestLabelFromDetectionAsync(It.IsAny<string>()))
                .ReturnsAsync(response);

            ActionResult<InterestLabelRemovalResponse> actionResult =
                await _controller.RemoveInterestLabelFromDetectionAsync("id");

            var contentResult = actionResult.Result as ObjectResult;
            Assert.IsNotNull(contentResult);
            Assert.AreEqual(200, contentResult.StatusCode);

            var result = (InterestLabelRemovalResponse)contentResult.Value!;

            Assert.IsNotNull(result);
            Assert.AreEqual(labelToRemove, result.LabelRemoved);

            _orchestrationServiceMock.Verify(service =>
                service.RemoveInterestLabelFromDetectionAsync(It.IsAny<string>()),
                Times.Once);
        }
    }
}
