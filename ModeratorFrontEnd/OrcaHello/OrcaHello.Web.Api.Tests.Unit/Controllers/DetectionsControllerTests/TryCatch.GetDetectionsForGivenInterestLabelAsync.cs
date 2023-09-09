namespace OrcaHello.Web.Api.Tests.Unit.Controllers
{
    public partial class DetectionsControllerTests
    {
        [TestMethod]
        public async Task TryCatch_GetDetectionsForGivenInterestLabelAsync_Expect_Exception()
        {
            _orchestrationServiceMock
                .SetupSequence(p => p.RetrieveDetectionsForGivenInterestLabelAsync(It.IsAny<string>()))

                .Throws(new DetectionOrchestrationValidationException(new Exception()))
                .Throws(new DetectionOrchestrationDependencyValidationException(new Exception()))

                .Throws(new DetectionOrchestrationDependencyException(new Exception()))
                .Throws(new DetectionOrchestrationServiceException(new Exception()))

                .Throws(new Exception());

            await ExecuteRetrieveDetectionsByLabel(2, StatusCodes.Status400BadRequest);
            await ExecuteRetrieveDetectionsByLabel(3, StatusCodes.Status500InternalServerError);

            _orchestrationServiceMock
                 .Verify(service => service
                    .RetrieveDetectionsForGivenInterestLabelAsync(It.IsAny<string>()),
                    Times.Exactly(5));

        }

        private async Task ExecuteRetrieveDetectionsByLabel(int count, int statusCode)
        {
            for (int x = 0; x < count; x++)
            {
                ActionResult<DetectionListForInterestLabelResponse> actionResult =
                    await _controller.GetDetectionsForGivenInterestLabelAsync(It.IsAny<string>());

                var contentResult = actionResult.Result as ObjectResult;
                Assert.IsNotNull(contentResult);
                Assert.AreEqual(statusCode, contentResult.StatusCode);
            }
        }
    }
}