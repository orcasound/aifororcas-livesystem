using static System.Runtime.InteropServices.JavaScript.JSType;

namespace OrcaHello.Web.UI.Tests.Unit.Services
{
    public partial class TagServiceTests
    {
        [TestMethod]
        public async Task Default_Expect_RetrieveFilteredTagsAsync()
        {
            TagListForTimeframeResponse expectedResponse = new()
            {
                Tags = new()
                {
                    "Tag 1",
                    "Tag 2"
                }
            };

            _apiBrokerMock.Setup(broker =>
                    broker.GetFilteredTagsAsync(It.IsAny<string>()))
                    .ReturnsAsync(expectedResponse);

            var result = await _service.RetrieveFilteredTagsAsync(DateTime.UtcNow.AddDays(-14), DateTime.UtcNow);

            Assert.AreEqual(expectedResponse.Tags.Count(), result.Tags.Count());

            _apiBrokerMock.Verify(broker =>
                broker.GetFilteredTagsAsync(It.IsAny<string>()),
                Times.Once);
        }
    }
}