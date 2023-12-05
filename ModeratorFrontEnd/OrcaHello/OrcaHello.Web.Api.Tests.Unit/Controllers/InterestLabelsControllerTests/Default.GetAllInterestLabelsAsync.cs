namespace OrcaHello.Web.Api.Tests.Unit.Controllers
{
    public partial class InterestLabelsControllerTests
    {
        [TestMethod]
        public async Task Default_GetAllInterestLabelsAsync_Expect_TagRemovalResponse()
        {
            InterestLabelListResponse response = new()
            {
                InterestLabels = new List<string>() { "Label1", "Label2" },
                Count = 2
            };

            _orchestrationServiceMock.Setup(service =>
                service.RetrieveAllInterestLabelsAsync())
                .ReturnsAsync(response);

            ActionResult<InterestLabelListResponse> actionResult =
                await _controller.GetAllInterestLabelsAsync();

            var contentResult = actionResult.Result as ObjectResult;
            Assert.IsNotNull(contentResult);
            Assert.AreEqual(200, contentResult.StatusCode);

            var result = (InterestLabelListResponse)contentResult.Value!;

            Assert.IsNotNull(result);
            Assert.AreEqual(2, result.Count);

            _orchestrationServiceMock.Verify(service =>
                service.RetrieveAllInterestLabelsAsync(),
                Times.Once);
        }
    }
}
