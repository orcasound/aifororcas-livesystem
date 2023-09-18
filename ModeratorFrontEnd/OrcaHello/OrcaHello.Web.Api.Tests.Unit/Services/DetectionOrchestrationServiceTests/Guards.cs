namespace OrcaHello.Web.Api.Tests.Unit.Services
{
    public partial class DetectionOrchestrationServiceTests
    {
        [TestMethod]
        public void Guard_AllGuardConditions_Expect_Exception()
        {
            var wrapper = new DetectionOrchestrationServiceWrapper();

            DateTime? invalidDate = DateTime.MinValue;

            Assert.ThrowsException<InvalidDetectionOrchestrationException>(() =>
                wrapper.Validate(invalidDate, nameof(invalidDate)));

            DateTime? nullDate = null;

            Assert.ThrowsException<InvalidDetectionOrchestrationException>(() =>
                wrapper.Validate(nullDate, nameof(nullDate)));

            string invalidProperty = string.Empty;

            Assert.ThrowsException<InvalidDetectionOrchestrationException>(() =>
                wrapper.Validate(invalidProperty, nameof(invalidProperty)));

            int invalidPage = 0;

            Assert.ThrowsException<InvalidDetectionOrchestrationException>(() =>
                wrapper.ValidatePage(invalidPage));

            int invalidPageSize = 0;
            Assert.ThrowsException<InvalidDetectionOrchestrationException>(() =>
                wrapper.ValidatePageSize(invalidPageSize));

            Metadata nullMetadata = null;

            Assert.ThrowsException<NotFoundMetadataException>(() =>
                wrapper.ValidateStorageMetadata(nullMetadata, Guid.NewGuid().ToString()));

            string invalidState = "Goober";

            Assert.ThrowsException<InvalidDetectionOrchestrationException>(() =>
                wrapper.ValidateStateIsAcceptable(invalidState));

            bool notDeleted = false;

            Assert.ThrowsException<DetectionNotDeletedException>(() =>
                wrapper.ValidateDeleted(notDeleted, "id"));

            bool notInserted = false;

            Assert.ThrowsException<DetectionNotInsertedException>(() =>
                wrapper.ValidateInserted(notInserted, "id"));

            ModerateDetectionRequest nullRequest = null;

            Assert.ThrowsException<NullModerateDetectionRequestException>(() =>
                wrapper.ValidateModerateRequestOnUpdate(nullRequest));

            ModerateDetectionRequest invalidRequest = new()
            {
                State = string.Empty,
                Moderator = string.Empty
            };

            Assert.ThrowsException<InvalidDetectionOrchestrationException>(() =>
                wrapper.ValidateModerateRequestOnUpdate(invalidRequest));

            invalidRequest.State = DetectionState.Positive.ToString();

            Assert.ThrowsException<InvalidDetectionOrchestrationException>(() =>
                wrapper.ValidateModerateRequestOnUpdate(invalidRequest));
        }
    }
}
