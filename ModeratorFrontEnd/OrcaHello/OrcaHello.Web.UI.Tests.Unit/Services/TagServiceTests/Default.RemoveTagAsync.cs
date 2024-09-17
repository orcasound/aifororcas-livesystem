using Moq;

namespace OrcaHello.Web.UI.Tests.Unit.Services
{
    public partial class TagServiceTests
    {
        [TestMethod]
        public async Task Default_Expect_RemoveTagAsync()
        {
            TagRemovalResponse expectedResponse = new()
            {
                Tag = "Tag 1",
                TotalMatching = 1,
                TotalRemoved = 1,
            };

            _apiBrokerMock.Setup(broker =>
                broker.RemoveTag(It.IsAny<string>()))
                .ReturnsAsync(expectedResponse);

            TagRemovalResponse actualResponse = await _service.RemoveTagAsync("Tag 1");

            Assert.AreEqual(expectedResponse, actualResponse);

            _apiBrokerMock.Verify(broker =>
                broker.RemoveTag(It.IsAny<string>()),
                Times.Once);
        }
    }
}