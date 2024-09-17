namespace OrcaHello.Web.Api.Tests.Unit.Controllers
{
    public partial class ModeratorsControllerTests
    {
        [TestMethod]
        public async Task Default_GetMetricsAsync_Expect_MetricsForModeratorResponse()
        {
            MetricsForModeratorResponse response = new()
            {
                FromDate = DateTime.Now,
                ToDate = DateTime.Now.AddDays(1),
                Positive = 1,
                Negative = 3,
                Unknown = 5,
                Moderator = "Moderator"
            };

            _orchestrationServiceMock.Setup(service =>
                service.RetrieveMetricsForGivenTimeframeAndModeratorAsync(It.IsAny<DateTime>(), It.IsAny<DateTime>(), It.IsAny<string>()))
                .ReturnsAsync(response);

            ActionResult<MetricsForModeratorResponse> actionResult =
                await _controller.GetMetricsForGivenTimeframeAndModeratorAsync("Moderator", DateTime.Now, DateTime.Now.AddDays(1));

            var contentResult = actionResult.Result as ObjectResult;
            Assert.IsNotNull(contentResult);
            Assert.AreEqual(200, contentResult.StatusCode);

            Assert.AreEqual(response.Positive,
                ((MetricsForModeratorResponse)contentResult.Value!).Positive);

            _orchestrationServiceMock.Verify(service =>
                service.RetrieveMetricsForGivenTimeframeAndModeratorAsync(It.IsAny<DateTime>(), It.IsAny<DateTime>(), It.IsAny<string>()),
                Times.Once);
        }
    }
}
