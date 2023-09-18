namespace OrcaHello.Web.Api.Tests.Unit.Controllers
{
    public partial class MetricsControllerTests
    {
        [TestMethod]
        public async Task TryCatch_GetMetricsAsync_Expect_Exception()
        {
            _orchestrationServiceMock
                .SetupSequence(p => p.RetrieveMetricsForGivenTimeframeAsync(It.IsAny<DateTime>(), It.IsAny<DateTime>()))

                .Throws(new MetricOrchestrationValidationException(new Exception()))
                .Throws(new MetricOrchestrationDependencyValidationException(new Exception()))

                .Throws(new MetricOrchestrationDependencyException(new Exception()))
                .Throws(new MetricOrchestrationServiceException(new Exception()))

                .Throws(new Exception());

            await ExecuteRetrieveMetrics(2, StatusCodes.Status400BadRequest);
            await ExecuteRetrieveMetrics(3, StatusCodes.Status500InternalServerError);

            _orchestrationServiceMock
                 .Verify(service => service
                    .RetrieveMetricsForGivenTimeframeAsync(It.IsAny<DateTime>(), It.IsAny<DateTime>()),
                    Times.Exactly(5));

        }

        private async Task ExecuteRetrieveMetrics(int count, int statusCode)
        {
            for (int x = 0; x < count; x++)
            {
                ActionResult<MetricsResponse> actionResult =
                    await _controller.GetMetricsAsync(DateTime.Now, DateTime.Now.AddDays(1));

                var contentResult = actionResult.Result as ObjectResult;
                Assert.IsNotNull(contentResult);
                Assert.AreEqual(statusCode, contentResult.StatusCode);
            }
        }
    }
}
