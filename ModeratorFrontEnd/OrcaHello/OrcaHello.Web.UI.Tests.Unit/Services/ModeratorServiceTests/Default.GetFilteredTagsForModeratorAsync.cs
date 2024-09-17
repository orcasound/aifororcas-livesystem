namespace OrcaHello.Web.UI.Tests.Unit.Services
{
    public partial class ModeratorServiceTests
    {
        [TestMethod]
        public async Task Default_Expect_GetFilteredTagsForModeratorAsync()
        {
            TagListForModeratorResponse expectedResponse = new()
            {
                Tags = new() { "Tag 1", "Tag 2" },
                Moderator = "John Smith"
            };

            _apiBrokerMock.Setup(broker =>
                broker.GetFilteredTagsForModeratorAsync(It.IsAny<string>(), It.IsAny<string>()))
                .ReturnsAsync(expectedResponse);

            TagListForModeratorResponse actualResponse = await _service.GetFilteredTagsForModeratorAsync("John Smith", DateTime.UtcNow.AddDays(-14), DateTime.UtcNow);

            Assert.AreEqual(expectedResponse, actualResponse);

            _apiBrokerMock.Verify(broker =>
                broker.GetFilteredTagsForModeratorAsync(It.IsAny<string>(), It.IsAny<string>()),
                Times.Once);
        }
    }
}
