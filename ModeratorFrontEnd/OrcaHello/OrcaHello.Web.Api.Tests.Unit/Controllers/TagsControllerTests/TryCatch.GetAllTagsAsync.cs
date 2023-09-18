namespace OrcaHello.Web.Api.Tests.Unit.Controllers
{
    public partial class TagsControllerTests
    {
        [TestMethod]
        public async Task TryCatch_GetAllTagsAsync_Expect_Exception()
        {
            _orchestrationServiceMock
                .SetupSequence(p => p.RetrieveAllTagsAsync())

                .Throws(new TagOrchestrationValidationException(new Exception()))
                .Throws(new TagOrchestrationDependencyValidationException(new Exception()))

                .Throws(new TagOrchestrationDependencyException(new Exception()))
                .Throws(new TagOrchestrationServiceException(new Exception()))

                .Throws(new Exception());

            await ExecuteRetrieveAllTags(2, StatusCodes.Status400BadRequest);
            await ExecuteRetrieveAllTags(3, StatusCodes.Status500InternalServerError);

            _orchestrationServiceMock
                 .Verify(service => service
                    .RetrieveAllTagsAsync(),
                    Times.Exactly(5));

        }

        private async Task ExecuteRetrieveAllTags(int count, int statusCode)
        {
            for (int x = 0; x < count; x++)
            {
                ActionResult<TagListResponse> actionResult =
                    await _controller.GetAllTagsAsync();

                var contentResult = actionResult.Result as ObjectResult;
                Assert.IsNotNull(contentResult);
                Assert.AreEqual(statusCode, contentResult.StatusCode);
            }
        }
    }
}
