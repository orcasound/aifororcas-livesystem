﻿using static OrcaHello.Web.UI.Services.HydrophoneViewService;

namespace OrcaHello.Web.UI.Tests.Unit.Services
{
    public partial class HydrophoneViewServiceTests
    {
        [TestMethod]
        public void TryCatch_ReturningGenericFunction_Expect_Exception()
        {
            var wrapper = new HydrophoneViewServiceWrapper();
            var delegateMock = new Mock<ReturningGenericFunction<List<HydrophoneItemView>>>();

            delegateMock
                .SetupSequence(p => p())

                .Throws(new NullHydrophoneViewResponseException("ResponseType"))
                .Throws(new InvalidHydrophoneViewException())

                .Throws(new HydrophoneValidationException())
                .Throws(new HydrophoneDependencyValidationException())

                .Throws(new HydrophoneDependencyException())
                .Throws(new HydrophoneServiceException())

                .Throws(new Exception());

            for (int x = 0; x < 2; x++)
                Assert.ThrowsExceptionAsync<HydrophoneViewValidationException>(async () =>
                    await wrapper.TryCatch(delegateMock.Object));

            for (int x = 0; x < 2; x++)
                Assert.ThrowsExceptionAsync<HydrophoneViewDependencyValidationException>(async () =>
                    await wrapper.TryCatch(delegateMock.Object));

            for (int x = 0; x < 2; x++)
                Assert.ThrowsExceptionAsync<HydrophoneViewDependencyException>(async () =>
                    await wrapper.TryCatch(delegateMock.Object));

            Assert.ThrowsExceptionAsync<HydrophoneViewServiceException>(async () =>
                await wrapper.TryCatch(delegateMock.Object));
        }
    }
}