namespace OrcaHello.Web.UI.Tests.Unit.Services
{
    public partial class MetricsServiceTest
    {
        [TestMethod]
        public async Task Default_Expect_RetrieveFilteredMetricsAsync()
        {
            MetricsResponse expectedResponse = new()
            {
                Unreviewed = 1,
                Positive = 2,
                Negative = 3,
                Unknown = 1
            };

            _apiBrokerMock.Setup(broker =>
                broker.GetFilteredMetricsAsync(It.IsAny<string>()))
                .ReturnsAsync(expectedResponse);

            MetricsResponse actualResponse = await _service.RetrieveFilteredMetricsAsync(DateTime.UtcNow.AddDays(-14), DateTime.UtcNow);

            Assert.AreEqual(expectedResponse, actualResponse);

            _apiBrokerMock.Verify(broker =>
                broker.GetFilteredMetricsAsync(It.IsAny<string>()),
                Times.Once);
        }
    }
}