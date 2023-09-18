namespace OrcaHello.Web.Api.Tests.Unit.Controllers
{
    public partial class InterestLabelsControllerTests
    {
        [TestMethod]
        public async Task TryCatch_GetAllInterestLabelsAsync_Expect_Exception()
        {
            _orchestrationServiceMock
                .SetupSequence(p => p.RetrieveAllInterestLabelsAsync())

                .Throws(new InterestLabelOrchestrationValidationException(new Exception()))
                .Throws(new InterestLabelOrchestrationDependencyValidationException(new Exception()))

                .Throws(new InterestLabelOrchestrationDependencyException(new Exception()))
                .Throws(new InterestLabelOrchestrationServiceException(new Exception()))

                .Throws(new Exception());

            await ExecuteRetrieveAllLabels(2, StatusCodes.Status400BadRequest);
            await ExecuteRetrieveAllLabels(3, StatusCodes.Status500InternalServerError);

            _orchestrationServiceMock
                 .Verify(service => service
                    .RetrieveAllInterestLabelsAsync(),
                    Times.Exactly(5));

        }

        private async Task ExecuteRetrieveAllLabels(int count, int statusCode)
        {
            for (int x = 0; x < count; x++)
            {
                ActionResult<InterestLabelListResponse> actionResult =
                    await _controller.GetAllInterestLabelsAsync();

                var contentResult = actionResult.Result as ObjectResult;
                Assert.IsNotNull(contentResult);
                Assert.AreEqual(statusCode, contentResult.StatusCode);
            }
        }
    }
}
