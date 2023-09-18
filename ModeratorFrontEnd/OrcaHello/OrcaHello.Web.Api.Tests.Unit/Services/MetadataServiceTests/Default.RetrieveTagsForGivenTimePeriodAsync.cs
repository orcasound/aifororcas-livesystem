namespace OrcaHello.Web.Api.Tests.Unit.Services
{
    public partial class MetadataServiceTests
    {
        [TestMethod]
        public async Task Default_Expect_RetrieveTagsForGivenTimePeriodAsync()
        {
            var tags = new List<string>
            {
                "Tag1",
                "Tag2"
            };

            _storageBrokerMock.Setup(broker =>
                broker.GetTagListByTimeframe(It.IsAny<DateTime>(), It.IsAny<DateTime>()))
                    .ReturnsAsync(tags);

            var result = await _metadataService.
                RetrieveTagsForGivenTimePeriodAsync(DateTime.Now, DateTime.Now.AddDays(1));

            Assert.AreEqual(tags.Count(), result.TotalCount);

            _storageBrokerMock.Verify(broker =>
                broker.GetTagListByTimeframe(It.IsAny<DateTime>(), It.IsAny<DateTime>()),
                Times.Once);
        }

    }
}