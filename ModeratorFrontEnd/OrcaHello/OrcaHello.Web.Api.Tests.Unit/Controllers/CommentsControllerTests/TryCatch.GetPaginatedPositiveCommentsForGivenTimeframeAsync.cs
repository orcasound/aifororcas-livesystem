namespace OrcaHello.Web.Api.Tests.Unit.Controllers
{
    public partial class CommentsControllerTests
    {
        [TestMethod]
        public async Task TryCatch_GetPaginatedPositiveCommentsForGivenTimeframeAsync_Expect_Exception()
        {
            _orchestrationServiceMock
                .SetupSequence(p => p.RetrievePositiveCommentsForGivenTimeframeAsync(It.IsAny<DateTime>(), It.IsAny<DateTime>(), It.IsAny<int>(), It.IsAny<int>()))

                .Throws(new CommentOrchestrationValidationException(new Exception()))
                .Throws(new CommentOrchestrationDependencyValidationException(new Exception()))

                .Throws(new CommentOrchestrationDependencyException(new Exception()))
                .Throws(new CommentOrchestrationServiceException(new Exception()))

                .Throws(new Exception());

            await ExecuteRetrieveComments(2, StatusCodes.Status400BadRequest);
            await ExecuteRetrieveComments(3, StatusCodes.Status500InternalServerError);

            _orchestrationServiceMock
                 .Verify(service => service
                    .RetrievePositiveCommentsForGivenTimeframeAsync(It.IsAny<DateTime>(), It.IsAny<DateTime>(), It.IsAny<int>(), It.IsAny<int>()),
                    Times.Exactly(5));

        }

        private async Task ExecuteRetrieveComments(int count, int statusCode)
        {
            for (int x = 0; x < count; x++)
            {
                ActionResult<CommentListResponse> actionResult =
                    await _controller.GetPaginatedPositiveCommentsForGivenTimeframeAsync(DateTime.Now, DateTime.Now.AddDays(1), 1, 10);

                var contentResult = actionResult.Result as ObjectResult;
                Assert.IsNotNull(contentResult);
                Assert.AreEqual(statusCode, contentResult.StatusCode);
            }
        }
    }
}
