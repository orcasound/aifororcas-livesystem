namespace OrcaHello.Web.Api.Tests.Unit.Controllers
{
    public partial class InterestLabelsControllerTests
    {
        [TestMethod]
        public async Task TryCatch_AddInterestLabelToDetectionAsync_Expect_Exception()
        {
            _orchestrationServiceMock
                .SetupSequence(p => p.AddInterestLabelToDetectionAsync(It.IsAny<string>(), It.IsAny<string>()))

                .Throws(new InterestLabelOrchestrationValidationException(new NotFoundMetadataException("id")))

                .Throws(new InterestLabelOrchestrationValidationException(new Exception()))
                .Throws(new InterestLabelOrchestrationDependencyValidationException(new Exception()))

                .Throws(new InterestLabelOrchestrationDependencyException(new Exception()))
                .Throws(new InterestLabelOrchestrationServiceException(new Exception()))

                .Throws(new Exception());


            await ExecuteAddLabel(1, StatusCodes.Status404NotFound);
            await ExecuteAddLabel(2, StatusCodes.Status400BadRequest);
            await ExecuteAddLabel(3, StatusCodes.Status500InternalServerError);

            _orchestrationServiceMock
                 .Verify(service => service
                    .AddInterestLabelToDetectionAsync(It.IsAny<string>(), It.IsAny<string>()),
                    Times.Exactly(6));

        }

        private async Task ExecuteAddLabel(int count, int statusCode)
        {
            for (int x = 0; x < count; x++)
            {
                ActionResult<InterestLabelAddResponse> actionResult =
                    await _controller.AddInterestLabelToDetectionAsync("id", "label");

                var contentResult = actionResult.Result as ObjectResult;
                Assert.IsNotNull(contentResult);
                Assert.AreEqual(statusCode, contentResult.StatusCode);
            }
        }

    }
}
