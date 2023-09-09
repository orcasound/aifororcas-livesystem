namespace OrcaHello.Web.Api.Tests.Unit.Services
{
    public partial class MetadataServiceTests
    {
        [TestMethod]
        public async Task Default_Expect_RetrieveAllTagsAsync()
        {
            var tags = new List<string>
            {
                "Tag1",
                "Tag2"
            };

            _storageBrokerMock.Setup(broker =>
                broker.GetAllTagList())
                    .ReturnsAsync(tags);

            var result = await _metadataService.
                RetrieveAllTagsAsync();

            Assert.AreEqual(tags.Count(), result.QueryableRecords.Count());

            _storageBrokerMock.Verify(broker =>
            broker.GetAllTagList(),
                Times.Once);
        }
    }
}
