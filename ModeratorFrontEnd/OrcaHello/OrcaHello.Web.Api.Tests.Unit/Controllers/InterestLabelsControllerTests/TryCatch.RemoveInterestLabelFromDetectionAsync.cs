namespace OrcaHello.Web.Api.Tests.Unit.Controllers
{
    public partial class InterestLabelsControllerTests
    {
        [TestMethod]
        public async Task TryCatch_RemoveInterestLabelFromDetectionAsync_Expect_Exception()
        {
            _orchestrationServiceMock
                .SetupSequence(p => p.RemoveInterestLabelFromDetectionAsync(It.IsAny<string>()))

                .Throws(new InterestLabelOrchestrationValidationException(new NotFoundMetadataException("id")))

                .Throws(new InterestLabelOrchestrationValidationException(new DetectionNotDeletedException("id")))
                .Throws(new InterestLabelOrchestrationValidationException(new DetectionNotInsertedException("id")))

                .Throws(new InterestLabelOrchestrationValidationException(new Exception()))
                .Throws(new InterestLabelOrchestrationDependencyValidationException(new Exception()))

                .Throws(new InterestLabelOrchestrationDependencyException(new Exception()))
                .Throws(new InterestLabelOrchestrationServiceException(new Exception()))
            .Throws(new Exception());

            await ExecuteRemoveLabel(1, StatusCodes.Status404NotFound);
            await ExecuteRemoveLabel(2, StatusCodes.Status422UnprocessableEntity);
            await ExecuteRemoveLabel(2, StatusCodes.Status400BadRequest);
            await ExecuteRemoveLabel(3, StatusCodes.Status500InternalServerError);

            _orchestrationServiceMock
                 .Verify(service => service
                    .RemoveInterestLabelFromDetectionAsync(It.IsAny<string>()),
                    Times.Exactly(8));

        }

        private async Task ExecuteRemoveLabel(int count, int statusCode)
        {
            for (int x = 0; x < count; x++)
            {
                ActionResult<InterestLabelRemovalResponse> actionResult =
                    await _controller.RemoveInterestLabelFromDetectionAsync("id");

                var contentResult = actionResult.Result as ObjectResult;
                Assert.IsNotNull(contentResult);
                Assert.AreEqual(statusCode, contentResult.StatusCode);
            }
        }
    }
}
