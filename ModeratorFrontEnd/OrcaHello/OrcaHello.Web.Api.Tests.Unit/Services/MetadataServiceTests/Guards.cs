using OrcaHello.Web.Api.Models;
using OrcaHello.Web.Api.Tests.Unit.Services.Metadatas;

namespace OrcaHello.Web.Api.Tests.Unit.Services
{
    public partial class MetadataServiceTests
    {
        [TestMethod]
        public void Guard_AllGuardConditions_Expect_Exception()
        {
            var wrapper = new MetadataServiceWrapper();

            DateTime invalidDate = DateTime.MinValue;

            Assert.ThrowsException<InvalidMetadataException>(() =>
                wrapper.Validate(invalidDate, nameof(invalidDate)));

            DateTime invalidFromDate = DateTime.Now;
            DateTime invalidToDate = DateTime.Now.AddDays(-10);

            Assert.ThrowsException<InvalidMetadataException>(() =>
                wrapper.ValidateDatesAreWithinRange(invalidFromDate, invalidToDate));

            string invalidTag = string.Empty;

            Assert.ThrowsException<InvalidMetadataException>(() =>
                wrapper.Validate(invalidTag, nameof(invalidTag)));

            Metadata nullMetadata = null!;

            Assert.ThrowsException<NullMetadataException>(() =>
                wrapper.ValidateMetadataOnCreate(nullMetadata));

            Assert.ThrowsException<NullMetadataException>(() =>
                wrapper.ValidateMetadataOnUpdate(nullMetadata));

            Metadata invalidMetadata = new()
            {
                Id = string.Empty,
                State = string.Empty,
                LocationName = string.Empty
            };

            Assert.ThrowsException<InvalidMetadataException>(() =>
                wrapper.ValidateMetadataOnCreate(invalidMetadata));

            Assert.ThrowsException<InvalidMetadataException>(() =>
                wrapper.ValidateMetadataOnUpdate(invalidMetadata));

            invalidMetadata.Id = Guid.NewGuid().ToString();

            Assert.ThrowsException<InvalidMetadataException>(() =>
                wrapper.ValidateMetadataOnCreate(invalidMetadata));

            Assert.ThrowsException<InvalidMetadataException>(() =>
                wrapper.ValidateMetadataOnUpdate(invalidMetadata));

            invalidMetadata.State = "Positive";

            Assert.ThrowsException<InvalidMetadataException>(() =>
                wrapper.ValidateMetadataOnCreate(invalidMetadata));

            Assert.ThrowsException<InvalidMetadataException>(() =>
                wrapper.ValidateMetadataOnUpdate(invalidMetadata));

            string invalidState = "Goober";

            Assert.ThrowsException<InvalidMetadataException>(() =>
                wrapper.ValidateStateIsAcceptable(invalidState));

            string invalidTags = "Goober;Goober2";

            Assert.ThrowsException<InvalidMetadataException>(() =>
                wrapper.ValidateTagContainsOnlyValidCharacters(invalidTags));

            string invalidConjunctions = "Goober,Goober2|Goober3";

            Assert.ThrowsException<InvalidMetadataException>(() =>
                wrapper.ValidateTagContainsOnlyValidCharacters(invalidConjunctions));
        }
    }
}
