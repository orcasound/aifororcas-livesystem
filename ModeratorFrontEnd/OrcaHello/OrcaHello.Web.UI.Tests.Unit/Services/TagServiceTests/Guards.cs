namespace OrcaHello.Web.UI.Tests.Unit.Services
{
    public partial class TagServiceTests
    {
        [TestMethod]
        public void Guard_AllGuardConditions_Expect_Exception()
        {
            var wrapper = new TagServiceWrapper();

            DateTime? invalidDate = DateTime.UtcNow.AddDays(1);
            DateTime? validDate = DateTime.UtcNow;

            Assert.ThrowsException<InvalidTagException>(() =>
                wrapper.ValidateDateRange(invalidDate, validDate));

            invalidDate = DateTime.UtcNow.AddDays(-1);

            Assert.ThrowsException<InvalidTagException>(() =>
                wrapper.ValidateDateRange(validDate, invalidDate));

            invalidDate = null;

            Assert.ThrowsException<InvalidTagException>(() =>
                wrapper.ValidateDateRange(invalidDate, validDate));

            Assert.ThrowsException<InvalidTagException>(() =>
                wrapper.ValidateDateRange(validDate, invalidDate));

            string invalidString = null!;

            Assert.ThrowsException<InvalidTagException>(() =>
                wrapper.Validate(invalidString, "invalidString"));

            ReplaceTagRequest? invalidRequest = null;

            Assert.ThrowsException<NullTagRequestException>(() =>
                wrapper.ValidateRequest(invalidRequest));

            TagListForTimeframeResponse? invalidResponse = null;

            Assert.ThrowsException<NullTagResponseException>(() =>
                wrapper.ValidateResponse(invalidResponse));
        }
    }
}
