namespace OrcaHello.Web.Api.Tests.Unit.Controllers
{
    public partial class TagsControllerTests
    {
        [TestMethod]
        public async Task TryCatch_DeleteTagFromAllDetectionsAsync_Expect_Exception()
        {
            _orchestrationServiceMock
                .SetupSequence(p => p.RemoveTagFromAllDetectionsAsync(It.IsAny<string>()))

                .Throws(new TagOrchestrationValidationException(new Exception()))
                .Throws(new TagOrchestrationDependencyValidationException(new Exception()))

                .Throws(new TagOrchestrationDependencyException(new Exception()))
                .Throws(new TagOrchestrationServiceException(new Exception()))

                .Throws(new Exception());

            await ExecuteRemoveTag(2, StatusCodes.Status400BadRequest);
            await ExecuteRemoveTag(3, StatusCodes.Status500InternalServerError);

            _orchestrationServiceMock
                 .Verify(service => service
                    .RemoveTagFromAllDetectionsAsync(It.IsAny<string>()),
                    Times.Exactly(5));

        }

        private async Task ExecuteRemoveTag(int count, int statusCode)
        {
            for (int x = 0; x < count; x++)
            {
                ActionResult<TagRemovalResponse> actionResult =
                    await _controller.DeleteTagFromAllDetectionsAsync("tag");

                var contentResult = actionResult.Result as ObjectResult;
                Assert.IsNotNull(contentResult);
                Assert.AreEqual(statusCode, contentResult.StatusCode);
            }
        }
    }
}
