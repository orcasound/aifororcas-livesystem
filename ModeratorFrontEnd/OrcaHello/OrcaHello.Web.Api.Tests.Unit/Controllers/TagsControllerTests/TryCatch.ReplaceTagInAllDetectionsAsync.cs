namespace OrcaHello.Web.Api.Tests.Unit.Controllers
{
    public partial class TagsControllerTests
    {
        [TestMethod]
        public async Task TryCatch_ReplaceTagInAllDetectionsAsync_Expect_Exception()
        {
            _orchestrationServiceMock
                .SetupSequence(p => p.ReplaceTagInAllDetectionsAsync(It.IsAny<ReplaceTagRequest>()))

                .Throws(new TagOrchestrationValidationException(new Exception()))
                .Throws(new TagOrchestrationDependencyValidationException(new Exception()))

                .Throws(new TagOrchestrationDependencyException(new Exception()))
                .Throws(new TagOrchestrationServiceException(new Exception()))

                .Throws(new Exception());

            await ExecuteReplaceTag(2, StatusCodes.Status400BadRequest);
            await ExecuteReplaceTag(3, StatusCodes.Status500InternalServerError);

            _orchestrationServiceMock
                 .Verify(service => service
                    .ReplaceTagInAllDetectionsAsync(It.IsAny<ReplaceTagRequest>()),
                    Times.Exactly(5));

        }

        private async Task ExecuteReplaceTag(int count, int statusCode)
        {
            for (int x = 0; x < count; x++)
            {
                ActionResult<TagReplaceResponse> actionResult =
                    await _controller.ReplaceTagInAllDetectionsAsync(new ReplaceTagRequest { OldTag = "oldTag", NewTag = "newTag" });

                var contentResult = actionResult.Result as ObjectResult;
                Assert.IsNotNull(contentResult);
                Assert.AreEqual(statusCode, contentResult.StatusCode);
            }
        }
    }
}
