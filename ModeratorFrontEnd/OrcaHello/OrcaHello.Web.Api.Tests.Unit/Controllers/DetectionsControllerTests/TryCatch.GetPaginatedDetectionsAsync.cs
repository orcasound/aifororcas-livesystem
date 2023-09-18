namespace OrcaHello.Web.Api.Tests.Unit.Controllers
{
    public partial class DetectionsControllerTests
    {


        [TestMethod]
        public async Task TryCatch_GetPaginatedDetectionsAsync_Expect_Exception()
        {
            _orchestrationServiceMock
                .SetupSequence(p => p.RetrieveFilteredDetectionsAsync(It.IsAny<DateTime>(), It.IsAny<DateTime>(), It.IsAny<string>(),
                It.IsAny<string>(), It.IsAny<bool>(), It.IsAny<string>(), It.IsAny<int>(), It.IsAny<int>()))

                .Throws(new DetectionOrchestrationValidationException(new Exception()))
                .Throws(new DetectionOrchestrationDependencyValidationException(new Exception()))

                .Throws(new DetectionOrchestrationDependencyException(new Exception()))
                .Throws(new DetectionOrchestrationServiceException(new Exception()))

                .Throws(new Exception());

            await ExecuteRetrieveDetections(2, StatusCodes.Status400BadRequest);
            await ExecuteRetrieveDetections(3, StatusCodes.Status500InternalServerError);

            _orchestrationServiceMock
                 .Verify(service => service
                    .RetrieveFilteredDetectionsAsync(It.IsAny<DateTime>(), It.IsAny<DateTime>(), It.IsAny<string>(),
                        It.IsAny<string>(), It.IsAny<bool>(), It.IsAny<string>(), It.IsAny<int>(), It.IsAny<int>()),
                    Times.Exactly(5));

        }

        private async Task ExecuteRetrieveDetections(int count, int statusCode)
        {
            for (int x = 0; x < count; x++)
            {
                ActionResult<DetectionListResponse> actionResult =
                    await _controller.GetPaginatedDetectionsAsync("state", DateTime.Now, DateTime.Now.AddDays(1), "sortBy", true,
                        1, 10, "location");

                var contentResult = actionResult.Result as ObjectResult;
                Assert.IsNotNull(contentResult);
                Assert.AreEqual(statusCode, contentResult.StatusCode);
            }
        }
    }
}
