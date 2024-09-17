namespace OrcaHello.Web.Api.Tests.Unit.Controllers
{
    public partial class DetectionsControllerTests
    {
        [TestMethod]
        public async Task TryCatch_PutModeratedInfoAsync_Expect_Exception()
        {
            _orchestrationServiceMock
                .SetupSequence(p => p.ModerateDetectionsByIdAsync(It.IsAny<ModerateDetectionsRequest>()))

                .Throws(new DetectionOrchestrationValidationException(new NotFoundMetadataException("id")))

                .Throws(new DetectionOrchestrationValidationException(new DetectionNotDeletedException("id")))
                .Throws(new DetectionOrchestrationValidationException(new DetectionNotInsertedException("id")))

                .Throws(new DetectionOrchestrationValidationException(new Exception()))
                .Throws(new DetectionOrchestrationDependencyValidationException(new Exception()))

                .Throws(new DetectionOrchestrationDependencyException(new Exception()))
                .Throws(new DetectionOrchestrationServiceException(new Exception()))

                .Throws(new Exception());

            await ExecuteModerateDetectionById(1, StatusCodes.Status404NotFound);
            await ExecuteModerateDetectionById(2, StatusCodes.Status422UnprocessableEntity);
            await ExecuteModerateDetectionById(2, StatusCodes.Status400BadRequest);
            await ExecuteModerateDetectionById(3, StatusCodes.Status500InternalServerError);

            _orchestrationServiceMock
                 .Verify(service => service
                    .ModerateDetectionsByIdAsync(It.IsAny<ModerateDetectionsRequest>()),
                    Times.Exactly(8));
        }

        private async Task ExecuteModerateDetectionById(int count, int statusCode)
        {
            for (int x = 0; x < count; x++)
            {
                ActionResult<ModerateDetectionsResponse> actionResult =
                    await _controller.PutModeratedInfoAsync(new ModerateDetectionsRequest());

                var contentResult = actionResult.Result as ObjectResult;
                Assert.IsNotNull(contentResult);
                Assert.AreEqual(statusCode, contentResult.StatusCode);
            }
        }
    }
}
