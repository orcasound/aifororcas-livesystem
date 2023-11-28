namespace OrcaHello.Web.UI.Tests.Unit.Services
{
    public partial class ModeratorServiceTests
    {
        [TestMethod]
        public void Guard_AllGuardConditions_Expect_Exception()
        {
            var wrapper = new ModeratorServiceWrapper();

            DateTime? invalidDate = DateTime.UtcNow.AddDays(1);
            DateTime? validDate = DateTime.UtcNow;

            Assert.ThrowsException<InvalidModeratorException>(() =>
                wrapper.ValidateDateRange(invalidDate, validDate));

            invalidDate = DateTime.UtcNow.AddDays(-1);

            Assert.ThrowsException<InvalidModeratorException>(() =>
                wrapper.ValidateDateRange(validDate, invalidDate));

            invalidDate = null;

            Assert.ThrowsException<InvalidModeratorException>(() =>
                wrapper.ValidateDateRange(invalidDate, validDate));

            Assert.ThrowsException<InvalidModeratorException>(() =>
                wrapper.ValidateDateRange(validDate, invalidDate));

            int invalidPage = -1;
            int validPageSize = 10;

            Assert.ThrowsException<InvalidModeratorException>(() =>
                wrapper.ValidatePagination(invalidPage, validPageSize));

            int validPage = 1;
            int invalidPageSize = -1;

            Assert.ThrowsException<InvalidModeratorException>(() =>
                wrapper.ValidatePagination(validPage, invalidPageSize));

            string invalidString = null!;

            Assert.ThrowsException<InvalidModeratorException>(() =>
                wrapper.Validate(invalidString, "invalidString"));

            MetricsResponse? invalidResponse = null;

            Assert.ThrowsException<NullModeratorResponseException>(() =>
                wrapper.ValidateResponse(invalidResponse));
        }
    }
}
