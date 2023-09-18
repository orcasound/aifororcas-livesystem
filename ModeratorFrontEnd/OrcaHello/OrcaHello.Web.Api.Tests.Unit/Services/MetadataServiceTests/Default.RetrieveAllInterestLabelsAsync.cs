namespace OrcaHello.Web.Api.Tests.Unit.Services
{
    public partial class MetadataServiceTests
    {
        [TestMethod]
        public async Task Default_Expect_RetrieveAllInterestLabelsAsync()
        {
            var labels = new List<string>
            {
                "Label1",
                "Label2"
            };

            _storageBrokerMock.Setup(broker =>
                broker.GetAllInterestLabels())
                    .ReturnsAsync(labels);

            var result = await _metadataService.
                RetrieveAllInterestLabelsAsync();

            Assert.AreEqual(labels.Count(), result.QueryableRecords.Count());

            _storageBrokerMock.Verify(broker =>
            broker.GetAllInterestLabels(),
                Times.Once);
        }
    }
}
