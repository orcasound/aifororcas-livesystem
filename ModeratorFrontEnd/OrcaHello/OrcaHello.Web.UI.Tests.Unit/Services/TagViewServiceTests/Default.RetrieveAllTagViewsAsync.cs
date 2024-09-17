namespace OrcaHello.Web.UI.Tests.Unit.Services
{
    public partial class TagViewServiceTests
    {
        [TestMethod]
        public async Task Default_Expect_RetrieveAllTagViewsAsync()
        {
            List<string> expectedResponse = new()
            {
                "Tag 1"
            };

            _tagServiceMock.Setup(broker =>
                broker.RetrieveAllTagsAsync())
                .ReturnsAsync(expectedResponse);

            List<TagItemView> actualResponse =
                await _viewService.RetrieveAllTagViewsAsync();

            Assert.AreEqual(expectedResponse.Count(), actualResponse.Count());

            _tagServiceMock.Verify(broker =>
                broker.RetrieveAllTagsAsync(),
                Times.Once);
        }
    }
}