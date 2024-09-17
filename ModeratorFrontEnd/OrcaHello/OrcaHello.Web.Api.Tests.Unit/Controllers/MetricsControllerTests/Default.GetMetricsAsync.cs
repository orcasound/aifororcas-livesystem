namespace OrcaHello.Web.Api.Tests.Unit.Controllers
{
    public partial class MetricsControllerTests
    {
        [TestMethod]
        public async Task Default_GetMetricsAsync_Expect_DetectionListResponse()
        {
            MetricsResponse response = new()
            {
                FromDate = DateTime.Now,
                ToDate = DateTime.Now.AddDays(1),
                Positive = 1,
                Negative = 3,
                Unknown = 5,
                Unreviewed = 10
            };

            _orchestrationServiceMock.Setup(service =>
                service.RetrieveMetricsForGivenTimeframeAsync(It.IsAny<DateTime>(), It.IsAny<DateTime>()))
                .ReturnsAsync(response);

            ActionResult<MetricsResponse> actionResult =
                await _controller.GetMetricsAsync(DateTime.Now, DateTime.Now.AddDays(1));

            var contentResult = actionResult.Result as ObjectResult;
            Assert.IsNotNull(contentResult);
            Assert.AreEqual(200, contentResult.StatusCode);

            Assert.AreEqual(response.Positive,
                ((MetricsResponse)contentResult.Value!).Positive);

            _orchestrationServiceMock.Verify(service =>
                service.RetrieveMetricsForGivenTimeframeAsync(It.IsAny<DateTime>(), It.IsAny<DateTime>()),
                Times.Once);
        }
    }
}