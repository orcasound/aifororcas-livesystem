namespace OrcaHello.Web.UI.Tests.Unit.Services
{
    public partial class TagViewServiceTests
    {
        [TestMethod]
        public async Task Default_Expect_DeleteTagAsync()
        {
            TagRemovalResponse expectedResponse = new()
            {
                Tag = "Tag 1",
                TotalMatching = 1,
                TotalRemoved = 1,
            };

            _tagServiceMock.Setup(broker =>
                broker.RemoveTagAsync(It.IsAny<string>()))
                .ReturnsAsync(expectedResponse);

            TagItemView request = new()
            {
                Tag = "Tag 1"
            };

            TagItemViewResponse actualResponse = await _viewService.DeleteTagAsync(request);

            Assert.AreEqual(expectedResponse.TotalMatching, actualResponse.MatchingTags);

            _tagServiceMock.Verify(broker =>
                broker.RemoveTagAsync(It.IsAny<string>()),
                Times.Once);
        }
    }
}