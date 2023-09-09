namespace OrcaHello.Web.Api.Tests.Unit.Controllers
{
    public partial class DetectionsControllerTests
    {
        [TestMethod]
        public async Task TryCatch_GetPaginatedDetectionsForGivenTimeframeAndTagAsync_Expect_Exception()
        {
            _orchestrationServiceMock
                .SetupSequence(p => p.RetrieveDetectionsForGivenTimeframeAndTagAsync(It.IsAny<DateTime>(), It.IsAny<DateTime>(), It.IsAny<string>(), It.IsAny<int>(), It.IsAny<int>()))

                .Throws(new DetectionOrchestrationValidationException(new Exception()))
                .Throws(new DetectionOrchestrationDependencyValidationException(new Exception()))

                .Throws(new DetectionOrchestrationDependencyException(new Exception()))
                .Throws(new DetectionOrchestrationServiceException(new Exception()))

                .Throws(new Exception());

            await ExecuteRetrieveDetectionsForTag(2, StatusCodes.Status400BadRequest);
            await ExecuteRetrieveDetectionsForTag(3, StatusCodes.Status500InternalServerError);

            _orchestrationServiceMock
                 .Verify(service => service
                    .RetrieveDetectionsForGivenTimeframeAndTagAsync(It.IsAny<DateTime>(), It.IsAny<DateTime>(), It.IsAny<string>(), It.IsAny<int>(), It.IsAny<int>()),
                    Times.Exactly(5));

        }

        private async Task ExecuteRetrieveDetectionsForTag(int count, int statusCode)
        {
            for (int x = 0; x < count; x++)
            {
                ActionResult<DetectionListForTagResponse> actionResult =
                    await _controller.GetPaginatedDetectionsForGivenTimeframeAndTagAsync("tag", DateTime.Now, DateTime.Now.AddDays(1), 1, 10);

                var contentResult = actionResult.Result as ObjectResult;
                Assert.IsNotNull(contentResult);
                Assert.AreEqual(statusCode, contentResult.StatusCode);
            }
        }
    }
}
