namespace OrcaHello.Web.UI.Tests.Unit.Services
{
    public partial class DashboardViewServiceTests
    {
        [TestMethod]
        public void Guard_AllGuardConditions_Expect_Exception()
        {
            var wrapper = new DashboardViewServiceWrapper();

            DateTime? invalidDate = DateTime.UtcNow.AddDays(1);
            DateTime? validDate = DateTime.UtcNow;

            Assert.ThrowsException<InvalidDashboardViewException>(() =>
                    wrapper.ValidateDateRange(invalidDate, validDate));

            invalidDate = DateTime.UtcNow.AddDays(-1);

            Assert.ThrowsException<InvalidDashboardViewException>(() =>
                wrapper.ValidateDateRange(validDate, invalidDate));

            invalidDate = null;

            Assert.ThrowsException<InvalidDashboardViewException>(() =>
                wrapper.ValidateDateRange(invalidDate, validDate));

            Assert.ThrowsException<InvalidDashboardViewException>(() =>
                wrapper.ValidateDateRange(validDate, invalidDate));

            int invalidPage = -1;
            int validPageSize = 10;

            Assert.ThrowsException<InvalidDashboardViewException>(() =>
                    wrapper.ValidatePagination(invalidPage, validPageSize));

            int validPage = 1;
            int invalidPageSize = -1;

            Assert.ThrowsException<InvalidDashboardViewException>(() =>
                    wrapper.ValidatePagination(validPage, invalidPageSize));

            string invalidModerator = null!;

            Assert.ThrowsException<InvalidDashboardViewException>(() =>
                wrapper.ValidateModerator(invalidModerator));

            string invalidTag = null!;

            Assert.ThrowsException<InvalidDashboardViewException>(() =>
                wrapper.ValidateTag(invalidTag));

            TagsByDateRequest? invalidRequest = null;

            Assert.ThrowsException<NullDashboardViewRequestException>(() =>
                wrapper.ValidateRequest(invalidRequest));

            DetectionListResponse? invalidResponse = null;

            Assert.ThrowsException<NullDashboardViewResponseException>(() =>
                wrapper.ValidateResponse(invalidResponse));

        }
    }
}