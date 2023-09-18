namespace OrcaHello.Web.Api.Tests.Unit.Controllers
{
    public partial class ModeratorsControllerTests
    {
        [TestMethod]
        public async Task TryCatch_CommentListForModeratorResponse_Positive_Expect_Exception()
        {
            _orchestrationServiceMock
                .SetupSequence(p => p.RetrievePositiveCommentsForGivenTimeframeAndModeratorAsync(It.IsAny<DateTime>(), It.IsAny<DateTime>(), It.IsAny<string>(), It.IsAny<int>(), It.IsAny<int>()))

                .Throws(new ModeratorOrchestrationValidationException(new Exception()))
                .Throws(new ModeratorOrchestrationDependencyValidationException(new Exception()))

                .Throws(new ModeratorOrchestrationDependencyException(new Exception()))
                .Throws(new ModeratorOrchestrationServiceException(new Exception()))

                .Throws(new Exception());

            await ExecuteRetrievePositiveComments(2, StatusCodes.Status400BadRequest);
            await ExecuteRetrievePositiveComments(3, StatusCodes.Status500InternalServerError);

            _orchestrationServiceMock
                 .Verify(service => service
                    .RetrievePositiveCommentsForGivenTimeframeAndModeratorAsync(It.IsAny<DateTime>(), It.IsAny<DateTime>(), It.IsAny<string>(), It.IsAny<int>(), It.IsAny<int>()),
                    Times.Exactly(5));

        }

        private async Task ExecuteRetrievePositiveComments(int count, int statusCode)
        {
            for (int x = 0; x < count; x++)
            {
                ActionResult<CommentListForModeratorResponse> actionResult =
                    await _controller.GetPaginatedPositiveCommentsForGivenTimeframeAndModeratorAsync("Moderator", DateTime.Now, DateTime.Now.AddDays(1), 1, 10);

                var contentResult = actionResult.Result as ObjectResult;
                Assert.IsNotNull(contentResult);
                Assert.AreEqual(statusCode, contentResult.StatusCode);
            }
        }

        [TestMethod]
        public async Task TryCatch_GetPaginatedNegativeAndUnknownCommentsForGivenTimeframeAndModeratorAsync_Expect_Exception()
        {
            _orchestrationServiceMock
                .SetupSequence(p => p.RetrieveNegativeAndUnknownCommentsForGivenTimeframeAndModeratorAsync(It.IsAny<DateTime>(), It.IsAny<DateTime>(), It.IsAny<string>(), It.IsAny<int>(), It.IsAny<int>()))

                .Throws(new ModeratorOrchestrationValidationException(new Exception()))
                .Throws(new ModeratorOrchestrationDependencyValidationException(new Exception()))

                .Throws(new ModeratorOrchestrationDependencyException(new Exception()))
                .Throws(new ModeratorOrchestrationServiceException(new Exception()))

                .Throws(new Exception());

            await ExecuteRetrieveNegativeAndUnknownComments(2, StatusCodes.Status400BadRequest);
            await ExecuteRetrieveNegativeAndUnknownComments(3, StatusCodes.Status500InternalServerError);

            _orchestrationServiceMock
                 .Verify(service => service
                    .RetrieveNegativeAndUnknownCommentsForGivenTimeframeAndModeratorAsync(It.IsAny<DateTime>(), It.IsAny<DateTime>(), It.IsAny<string>(), It.IsAny<int>(), It.IsAny<int>()),
                    Times.Exactly(5));

        }

        private async Task ExecuteRetrieveNegativeAndUnknownComments(int count, int statusCode)
        {
            for (int x = 0; x < count; x++)
            {
                ActionResult<CommentListForModeratorResponse> actionResult =
                    await _controller.GetPaginatedNegativeAndUnknownCommentsForGivenTimeframeAndModeratorAsync("Moderator", DateTime.Now, DateTime.Now.AddDays(1), 1, 10);

                var contentResult = actionResult.Result as ObjectResult;
                Assert.IsNotNull(contentResult);
                Assert.AreEqual(statusCode, contentResult.StatusCode);
            }
        }
    }
}
