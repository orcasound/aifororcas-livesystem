namespace OrcaHello.Web.UI.Tests.Unit.Services
{
    public partial class TagServiceTests
    {
        [TestMethod]
        public async Task Default_Expect_RetrieveAllTagsAsync()
        {
            TagListResponse expectedResponse = new()
            {
                Tags = new()
                {
                    "Tag 1",
                    "Tag 2"
                }
            };

            _apiBrokerMock.Setup(broker =>
                broker.GetAllTagsAsync())
                .ReturnsAsync(expectedResponse);

            var result = await _service.RetrieveAllTagsAsync();

            Assert.AreEqual(expectedResponse.Tags.Count(), result.Count());

            _apiBrokerMock.Verify(broker =>
                broker.GetAllTagsAsync(),
                Times.Once);
        }
    }
}