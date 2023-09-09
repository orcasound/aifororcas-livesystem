namespace OrcaHello.Web.Api.Tests.Unit.Services
{
    public partial class HydrophoneServiceTests
    {
        [TestMethod]
        public async Task Default_Expect_RetrieveAllHydrophonesAsync()
        {
            var hydrophones = new List<HydrophoneData>
            {
                new()
            };

            _hydrophoneBrokerMock.Setup(broker =>
                broker.GetFeedsAsync())
                    .ReturnsAsync(hydrophones);

            var result = await _hydrophoneService.
                RetrieveAllHydrophonesAsync();

            Assert.AreEqual(hydrophones.Count, result.QueryableRecords.Count());

            _hydrophoneBrokerMock.Verify(broker =>
            broker.GetFeedsAsync(),
                Times.Once);
        }
    }
}
