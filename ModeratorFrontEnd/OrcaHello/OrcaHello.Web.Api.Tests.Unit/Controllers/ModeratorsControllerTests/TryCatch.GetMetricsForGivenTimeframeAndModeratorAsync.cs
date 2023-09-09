namespace OrcaHello.Web.Api.Tests.Unit.Controllers
{
    public partial class ModeratorsControllerTests
    {
        [TestMethod]
        public async Task TryCatch_GetMetricsForGivenTimeframeAndModeratorAsync_Expect_Exception()
        {
            _orchestrationServiceMock
                .SetupSequence(p => p.RetrieveMetricsForGivenTimeframeAndModeratorAsync(It.IsAny<DateTime>(), It.IsAny<DateTime>(), It.IsAny<string>()))

                .Throws(new ModeratorOrchestrationValidationException(new Exception()))
                .Throws(new ModeratorOrchestrationDependencyValidationException(new Exception()))

                .Throws(new ModeratorOrchestrationDependencyException(new Exception()))
                .Throws(new ModeratorOrchestrationServiceException(new Exception()))

                .Throws(new Exception());

            await ExecuteRetrieveMetrics(2, StatusCodes.Status400BadRequest);
            await ExecuteRetrieveMetrics(3, StatusCodes.Status500InternalServerError);

            _orchestrationServiceMock
                 .Verify(service => service
                    .RetrieveMetricsForGivenTimeframeAndModeratorAsync(It.IsAny<DateTime>(), It.IsAny<DateTime>(), It.IsAny<string>()),
                    Times.Exactly(5));

        }

        private async Task ExecuteRetrieveMetrics(int count, int statusCode)
        {
            for (int x = 0; x < count; x++)
            {
                ActionResult<MetricsForModeratorResponse> actionResult =
                    await _controller.GetMetricsForGivenTimeframeAndModeratorAsync("Moderator", DateTime.Now, DateTime.Now.AddDays(1));

                var contentResult = actionResult.Result as ObjectResult;
                Assert.IsNotNull(contentResult);
                Assert.AreEqual(statusCode, contentResult.StatusCode);
            }
        }

    }
}
