namespace OrcaHello.Web.UI.Tests.Unit.Services
{
    public partial class CommentServiceTests
    {
        [TestMethod]
        public async Task Default_Expect_RetrieveFilteredNegativeAndUnknownCommentsAsync()
        {
            CommentListResponse expectedResponse = new()
            {
                Comments = new()
                {
                    new() { Id = Guid.NewGuid().ToString(), Comments = "Test comment." }
                }
            };

            _apiBrokerMock.Setup(broker =>
                broker.GetFilteredNegativeAndUknownCommentsAsync(It.IsAny<string>()))
                .ReturnsAsync(expectedResponse);

            var result = await _service.RetrieveFilteredNegativeAndUnknownCommentsAsync(DateTime.UtcNow.AddDays(-14), DateTime.UtcNow, 1, 10);

            Assert.AreEqual(expectedResponse.Comments.Count(), result.Comments.Count());

            _apiBrokerMock.Verify(broker =>
                broker.GetFilteredNegativeAndUknownCommentsAsync(It.IsAny<string>()),
                Times.Once);
        }
    }
}
