namespace OrcaHello.Web.UI.Tests.Unit.Services
{
    public partial class DetectionServiceTests
    {
        [TestMethod]
        public void Guard_AllGuardConditions_Expect_Exception()
        {
            var wrapper = new DetectionServiceWrapper();

            string invalidString = null!;

            Assert.ThrowsException<InvalidDetectionException>(() =>
                wrapper.Validate(invalidString, "invalidString"));

            DateTime? invalidDate = null!;

            Assert.ThrowsException<InvalidDetectionException>(() =>
                wrapper.Validate(invalidDate, "invalidDate"));

            invalidDate = DateTime.MinValue!;

            Assert.ThrowsException<InvalidDetectionException>(() =>
                wrapper.Validate(invalidDate, "invalidDate"));

            List<string> badIds = null!;

            Assert.ThrowsException<InvalidDetectionException>(() =>
                wrapper.ValidateAtLeastOneId(badIds, "badIds"));

            badIds = new List<string>();

            Assert.ThrowsException<InvalidDetectionException>(() =>
                wrapper.ValidateAtLeastOneId(badIds, "badIds"));

            badIds = new List<string>() { string.Empty };

            Assert.ThrowsException<InvalidDetectionException>(() =>
                wrapper.ValidateAtLeastOneId(badIds, "badIds"));

            invalidDate = DateTime.UtcNow.AddDays(1);
            DateTime? validDate = DateTime.UtcNow;

            Assert.ThrowsException<InvalidDetectionException>(() =>
                wrapper.ValidateDateRange(invalidDate, validDate));

            invalidDate = DateTime.UtcNow.AddDays(-1);

            Assert.ThrowsException<InvalidDetectionException>(() =>
                wrapper.ValidateDateRange(validDate, invalidDate));

            invalidDate = null;

            Assert.ThrowsException<InvalidDetectionException>(() =>
                wrapper.ValidateDateRange(invalidDate, validDate));

            Assert.ThrowsException<InvalidDetectionException>(() =>
                wrapper.ValidateDateRange(validDate, invalidDate));

            int invalidPage = -1;
            int validPageSize = 10;

            Assert.ThrowsException<InvalidDetectionException>(() =>
                wrapper.ValidatePagination(invalidPage, validPageSize));

            int validPage = 1;
            int invalidPageSize = -1;

            Assert.ThrowsException<InvalidDetectionException>(() =>
                wrapper.ValidatePagination(validPage, invalidPageSize));

            ModerateDetectionsRequest? invalidRequest = null;

            Assert.ThrowsException<NullDetectionRequestException>(() =>
                wrapper.ValidateRequest(invalidRequest));

            DetectionListResponse? invalidResponse = null;

            Assert.ThrowsException<NullDetectionResponseException>(() =>
                wrapper.ValidateResponse(invalidResponse));
        }
    }
}
