namespace OrcaHello.Web.UI.Tests.Unit.Services
{
    public partial class MetricsServiceTest
    {
        [TestMethod]
        public void Guard_AllGuardConditions_Expect_Exception()
        {
            var wrapper = new MetricsServiceWrapper();

            DateTime? invalidDate = DateTime.UtcNow.AddDays(1);
            DateTime? validDate = DateTime.UtcNow;

            Assert.ThrowsException<InvalidMetricsException>(() =>
                wrapper.ValidateDateRange(invalidDate, validDate));

            invalidDate = DateTime.UtcNow.AddDays(-1);

            Assert.ThrowsException<InvalidMetricsException>(() =>
                wrapper.ValidateDateRange(validDate, invalidDate));

            invalidDate = null;

            Assert.ThrowsException<InvalidMetricsException>(() =>
                wrapper.ValidateDateRange(invalidDate, validDate));

            Assert.ThrowsException<InvalidMetricsException>(() =>
                wrapper.ValidateDateRange(validDate, invalidDate));

            MetricsResponse? invalidResponse = null;

            Assert.ThrowsException<NullMetricsResponseException>(() =>
                wrapper.ValidateResponse(invalidResponse));
        }
    }
}
