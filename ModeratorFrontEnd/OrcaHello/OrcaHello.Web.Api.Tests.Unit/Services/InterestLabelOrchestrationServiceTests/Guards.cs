using OrcaHello.Web.Shared.Utilities;

namespace OrcaHello.Web.Api.Tests.Unit.Services
{
    public partial class InterestLabelOrchestrationServiceTests
    {
        [TestMethod]
        public void Guard_AllGuardConditions_Expect_Exception()
        {
            var wrapper = new InterestLabelOrchestrationServiceWrapper();

            string invalidProperty = string.Empty;

            Assert.ThrowsException<InvalidInterestLabelOrchestrationException>(() =>
                wrapper.Validate(invalidProperty, nameof(invalidProperty)));

            Metadata nullMetadata = null!;

            Assert.ThrowsException<NotFoundMetadataException>(() =>
                wrapper.ValidateMetadataFound(nullMetadata, "id"));

            bool notDeleted = false;

            Assert.ThrowsException<DetectionNotDeletedException>(() =>
                wrapper.ValidateDeleted(notDeleted, "id"));

            bool notInserted = false;

            Assert.ThrowsException<DetectionNotInsertedException>(() =>
                wrapper.ValidateInserted(notInserted, "id"));

        }
    }
}