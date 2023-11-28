namespace OrcaHello.Web.UI.Tests.Unit.Services
{
    public partial class ModeratorServiceTests
    {
        [TestMethod]
        public async Task Default_Expect_GetFilteredNegativeAndUnknownCommentsForModeratorAsync()
        {
            CommentListForModeratorResponse expectedResponse = new()
            {
                Comments = new()
                {
                    new() { Id = Guid.NewGuid().ToString(), Comments = "Test comment." }
                },
                Moderator = "John Smith"
            };

            _apiBrokerMock.Setup(broker =>
                broker.GetFilteredNegativeAndUknownCommentsForModeratorAsync(It.IsAny<string>(), It.IsAny<string>()))
                .ReturnsAsync(expectedResponse);

            var result = await _service.GetFilteredNegativeAndUknownCommentsForModeratorAsync("John Smith", DateTime.UtcNow.AddDays(-14), DateTime.UtcNow, 1, 10);

            Assert.AreEqual(expectedResponse.Comments.Count(), result.Comments.Count());

            _apiBrokerMock.Verify(broker =>
                broker.GetFilteredNegativeAndUknownCommentsForModeratorAsync(It.IsAny<string>(), It.IsAny<string>()),
                Times.Once);
        }
    }
}