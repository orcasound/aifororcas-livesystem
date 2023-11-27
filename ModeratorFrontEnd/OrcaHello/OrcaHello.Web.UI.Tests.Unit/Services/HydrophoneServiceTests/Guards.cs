namespace OrcaHello.Web.UI.Tests.Unit.Services
{
    public partial class HydrophoneServiceTests
    {
        [TestMethod]
        public void Guard_AllGuardConditions_Expect_Exception()
        {
            var wrapper = new HydrophoneServiceWrapper();

            int invalidCount = 0;

            Assert.ThrowsException<InvalidHydrophoneException>(() =>
                wrapper.ValidateThereAreHydrophones(invalidCount));

            HydrophoneListResponse? invalidResponse = null;

            Assert.ThrowsException<NullHydrophoneResponseException>(() =>
                wrapper.ValidateResponse(invalidResponse));
        }
    }
}
