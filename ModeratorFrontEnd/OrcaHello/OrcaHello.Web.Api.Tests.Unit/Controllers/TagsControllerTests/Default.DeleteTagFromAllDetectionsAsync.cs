namespace OrcaHello.Web.Api.Tests.Unit.Controllers
{
    public partial class TagsControllerTests
    {
        [TestMethod]
        public async Task Default_DeleteTagFromAllDetectionsAsync_Expect_TagRemovalResponse()
        {
            var tagToRemove = "tagToRemove";

            TagRemovalResponse response = new()
            {
                Tag = tagToRemove,
                TotalMatching = 2,
                TotalRemoved = 2
            };

            _orchestrationServiceMock.Setup(service =>
                service.RemoveTagFromAllDetectionsAsync(It.IsAny<string>()))
                .ReturnsAsync(response);

            ActionResult<TagRemovalResponse> actionResult =
                await _controller.DeleteTagFromAllDetectionsAsync(tagToRemove);

            var contentResult = actionResult.Result as ObjectResult;
            Assert.IsNotNull(contentResult);
            Assert.AreEqual(200, contentResult.StatusCode);

            var result = (TagRemovalResponse)contentResult.Value!;

            Assert.IsNotNull(result);
            Assert.AreEqual(tagToRemove, result.Tag);
            Assert.AreEqual(2, result.TotalMatching);
            Assert.AreEqual(2, result.TotalRemoved);

            _orchestrationServiceMock.Verify(service =>
                service.RemoveTagFromAllDetectionsAsync(It.IsAny<string>()),
                Times.Once);
        }
    }
}
