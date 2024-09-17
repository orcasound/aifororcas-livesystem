namespace OrcaHello.Web.UI.Tests.Unit.Services
{
    public partial class TagViewServiceTests
    {
        [TestMethod]
        public async Task Default_Expect_ReplaceTagAsync()
        {
            TagReplaceResponse expectedResponse = new()
            {
                OldTag = "OldTag",
                NewTag = "NewTag",
                TotalMatching = 1,
                TotalReplaced = 1
            };

            _tagServiceMock.Setup(broker =>
                broker.ReplaceTagAsync(It.IsAny<ReplaceTagRequest>()))
                .ReturnsAsync(expectedResponse);

            ReplaceTagRequest request = new()
            {
                OldTag = "OldTag",
                NewTag = "NewTag"
            };

            TagItemViewResponse actualResponse =
                await _viewService.ReplaceTagAsync(request);

            Assert.AreEqual(expectedResponse.TotalMatching, actualResponse.MatchingTags);

            _tagServiceMock.Verify(broker =>
                broker.ReplaceTagAsync(It.IsAny<ReplaceTagRequest>()),
                Times.Once);
        }
    }
}