namespace OrcaHello.Web.UI.Tests.Unit.Services
{
    public partial class HydrophoneViewServiceTests
    {
        [TestMethod]
        public void Guard_AllGuardConditions_Expect_Exception()
        {
            var wrapper = new HydrophoneViewServiceWrapper();

            List<Hydrophone> invalidResponse = null!;

            Assert.ThrowsException<NullHydrophoneViewResponseException>(() =>
                wrapper.ValidateResponse(invalidResponse));
        }
    }
}
