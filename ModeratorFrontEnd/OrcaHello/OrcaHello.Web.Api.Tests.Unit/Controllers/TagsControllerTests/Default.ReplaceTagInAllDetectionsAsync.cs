namespace OrcaHello.Web.Api.Tests.Unit.Controllers
{
    public partial class TagsControllerTests
    {
        [TestMethod]
        public async Task Default_ReplaceTagInAllDetectionsAsync_Expect_TagReplaceResponse()
        {
            var oldTag = "oldTag";
            var newTag = "newTag";

            TagReplaceResponse response = new()
            {
                OldTag = oldTag,
                NewTag = newTag,
                TotalMatching = 2,
                TotalReplaced = 2
            };

            var request = new ReplaceTagRequest
            {
                OldTag = oldTag,
                NewTag = newTag,
            };

            _orchestrationServiceMock.Setup(service =>
                service.ReplaceTagInAllDetectionsAsync(It.IsAny<ReplaceTagRequest>()))
                .ReturnsAsync(response);

            ActionResult<TagReplaceResponse> actionResult =
                await _controller.ReplaceTagInAllDetectionsAsync(request);

            var contentResult = actionResult.Result as ObjectResult;
            Assert.IsNotNull(contentResult);
            Assert.AreEqual(200, contentResult.StatusCode);

            var result = (TagReplaceResponse)contentResult.Value!;

            Assert.IsNotNull(result);
            Assert.AreEqual(oldTag, result.OldTag);
            Assert.AreEqual(newTag, result.NewTag);
            Assert.AreEqual(2, result.TotalMatching);
            Assert.AreEqual(2, result.TotalReplaced);

            _orchestrationServiceMock.Verify(service =>
                service.ReplaceTagInAllDetectionsAsync(It.IsAny<ReplaceTagRequest>()),
                Times.Once);
        }
    }
}
