namespace OrcaHello.Web.UI.Tests.Unit.Services
{
    public partial class TagServiceTests
    {
        [TestMethod]
        public async Task Default_Expect_ReplaceTagAsync()
        {
            TagReplaceResponse expectedResponse = new()
            {
                OldTag = "Tag 1",
                NewTag = "Tag 2",
                TotalMatching = 1,
                TotalReplaced = 1,
            };

            _apiBrokerMock.Setup(broker =>
                broker.ReplaceTagAsync(It.IsAny<ReplaceTagRequest>()))
                .ReturnsAsync(expectedResponse);

            TagReplaceResponse actualResponse = await _service.ReplaceTagAsync(new ReplaceTagRequest { OldTag = "Tag1", NewTag = "Tag2" });

            Assert.AreEqual(expectedResponse, actualResponse);

            _apiBrokerMock.Verify(broker =>
                broker.ReplaceTagAsync(It.IsAny<ReplaceTagRequest>()),
                Times.Once);
        }
    }
}