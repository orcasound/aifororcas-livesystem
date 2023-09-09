namespace OrcaHello.Web.Api.Tests.Unit.Controllers
{
    public partial class DetectionsControllerTests
    {
        [TestMethod]
        public async Task TryCatch_GetDetectionByIdAsync_Expect_Exception()
        {
            _orchestrationServiceMock
                .SetupSequence(p => p.RetrieveDetectionByIdAsync(It.IsAny<string>()))

                .Throws(new DetectionOrchestrationValidationException(new NotFoundMetadataException("id")))

                .Throws(new DetectionOrchestrationValidationException(new Exception()))
                .Throws(new DetectionOrchestrationDependencyValidationException(new Exception()))

                .Throws(new DetectionOrchestrationDependencyException(new Exception()))
                .Throws(new DetectionOrchestrationServiceException(new Exception()))

                .Throws(new Exception());

            await ExecuteRetrieveDetectionById(1, StatusCodes.Status404NotFound);
            await ExecuteRetrieveDetectionById(2, StatusCodes.Status400BadRequest);
            await ExecuteRetrieveDetectionById(3, StatusCodes.Status500InternalServerError);

            _orchestrationServiceMock
                 .Verify(service => service
                    .RetrieveDetectionByIdAsync(It.IsAny<string>()),
                    Times.Exactly(6));

        }

        private async Task ExecuteRetrieveDetectionById(int count, int statusCode)
        {
            for (int x = 0; x < count; x++)
            {
                ActionResult<Detection> actionResult =
                    await _controller.GetDetectionByIdAsync(It.IsAny<string>());

                var contentResult = actionResult.Result as ObjectResult;
                Assert.IsNotNull(contentResult);
                Assert.AreEqual(statusCode, contentResult.StatusCode);
            }
        }
    }
}
