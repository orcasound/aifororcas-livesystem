namespace OrcaHello.Web.UI.Tests.Unit.Services
{
    public partial class DetectionViewServiceTests
    {
        [TestMethod]
        public void Guard_AllGuardConditions_Expect_Exception()
        {
            var wrapper = new DetectionViewServiceWrapper();

            string invalidString = null!;

            Assert.ThrowsException<InvalidDetectionViewException>(() =>
                wrapper.Validate(invalidString, "invalidString"));

            List<string> badIds = null!;

            Assert.ThrowsException<InvalidDetectionViewException>(() =>
                wrapper.ValidateAtLeastOneId(badIds));

            badIds = new();

            Assert.ThrowsException<InvalidDetectionViewException>(() =>
                wrapper.ValidateAtLeastOneId(badIds));

            badIds = new() { string.Empty };

            Assert.ThrowsException<InvalidDetectionViewException>(() =>
                wrapper.ValidateAtLeastOneId(badIds));

            DateTime? invalidDate = DateTime.UtcNow.AddDays(1);
            DateTime? validDate = DateTime.UtcNow;

            Assert.ThrowsException<InvalidDetectionViewException>(() =>
                wrapper.ValidateDateRange(invalidDate, validDate));

            invalidDate = DateTime.UtcNow.AddDays(-1);

            Assert.ThrowsException<InvalidDetectionViewException>(() =>
                wrapper.ValidateDateRange(validDate, invalidDate));

            invalidDate = null;

            Assert.ThrowsException<InvalidDetectionViewException>(() =>
                wrapper.ValidateDateRange(invalidDate, validDate));

            Assert.ThrowsException<InvalidDetectionViewException>(() =>
                wrapper.ValidateDateRange(validDate, invalidDate));

            int invalidPage = -1;
            int validPageSize = 10;

            Assert.ThrowsException<InvalidDetectionViewException>(() =>
                wrapper.ValidatePagination(invalidPage, validPageSize));

            int validPage = 1;
            int invalidPageSize = -1;

            Assert.ThrowsException<InvalidDetectionViewException>(() =>
                wrapper.ValidatePagination(validPage, invalidPageSize));

            PaginatedDetectionsByStateRequest? invalidRequest = null;

            Assert.ThrowsException<NullDetectionViewRequestException>(() =>
                wrapper.ValidateRequest(invalidRequest));

            DetectionListResponse? invalidResponse = null;

            Assert.ThrowsException<NullDetectionViewResponseException>(() =>
                wrapper.ValidateResponse(invalidResponse));
        }
    }
}
