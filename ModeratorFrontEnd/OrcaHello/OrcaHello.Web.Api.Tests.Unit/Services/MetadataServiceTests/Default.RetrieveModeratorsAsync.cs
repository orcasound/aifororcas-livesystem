namespace OrcaHello.Web.Api.Tests.Unit.Services
{
    public partial class MetadataServiceTests
    {
        [TestMethod]
        public async Task Default_Expect_RetrieveModeratorsAsync()
        {
            List<string> expectedResults = new()
            {
                "Moderator 1",
                "Moderator 2"
            };

            _storageBrokerMock.Setup(broker =>
                broker.GetModeratorList())
                    .ReturnsAsync(expectedResults);

            var result = await _metadataService.
                RetrieveModeratorsAsync();

            Assert.AreEqual(expectedResults.Count(), result.QueryableRecords.Count());

            _storageBrokerMock.Verify(broker =>
                broker.GetModeratorList(),
                Times.Once);
        }
    }

}
