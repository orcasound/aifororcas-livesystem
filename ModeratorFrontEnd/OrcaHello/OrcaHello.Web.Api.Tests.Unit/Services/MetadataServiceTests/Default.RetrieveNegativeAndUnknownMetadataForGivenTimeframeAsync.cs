﻿using Moq;
using OrcaHello.Web.Api.Models;

namespace OrcaHello.Web.Api.Tests.Unit.Services
{
    public partial class MetadataServiceTests
    {
        [TestMethod]
        public async Task Default_Expect_RetrieveNegativeAndUnknownMetadataForGivenTimeframeAsync()
        {
            ListMetadataAndCount expectedResult = new()
            {
                PaginatedRecords = new()
                {
                    CreateRandomMetadata()
                },
                TotalCount = 1
            };

            _storageBrokerMock.Setup(broker =>
                broker.GetNegativeAndUnknownMetadataListByTimeframe(It.IsAny<DateTime>(), It.IsAny<DateTime>(), It.IsAny<int>(), It.IsAny<int>()))
                    .ReturnsAsync(expectedResult);

            DateTime fromDate = DateTime.Now;
            DateTime toDate = DateTime.Now.AddDays(1);

            var result = await _metadataService.
                RetrieveNegativeAndUnknownMetadataForGivenTimeframeAsync(fromDate, toDate, 1, 10);

            Assert.AreEqual(expectedResult.PaginatedRecords.Count(), result.QueryableRecords.Count());

            _storageBrokerMock.Verify(broker =>
                broker.GetNegativeAndUnknownMetadataListByTimeframe(It.IsAny<DateTime>(), It.IsAny<DateTime>(), It.IsAny<int>(), It.IsAny<int>()),
               Times.Once);
        }

        [TestMethod]
        public async Task Default_Expect_RetrieveNegativeAndUnknownMetadataForGivenTimeframeAsync_ZeroPageAndPageSize()
        {
            ListMetadataAndCount expectedResult = new()
            {
                PaginatedRecords = new()
                {
                    CreateRandomMetadata()
                },
                TotalCount = 1
            };

            _storageBrokerMock.Setup(broker =>
                broker.GetNegativeAndUnknownMetadataListByTimeframe(It.IsAny<DateTime>(), It.IsAny<DateTime>(), It.IsAny<int>(), It.IsAny<int>()))
                    .ReturnsAsync(expectedResult);

            DateTime fromDate = DateTime.Now;
            DateTime toDate = DateTime.Now.AddDays(1);

            var result = await _metadataService.
                RetrieveNegativeAndUnknownMetadataForGivenTimeframeAsync(fromDate, toDate, -1, -10);

            Assert.AreEqual(expectedResult.PaginatedRecords.Count(), result.QueryableRecords.Count());

            _storageBrokerMock.Verify(broker =>
                broker.GetNegativeAndUnknownMetadataListByTimeframe(It.IsAny<DateTime>(), It.IsAny<DateTime>(), It.IsAny<int>(), It.IsAny<int>()),
               Times.Once);
        }
    }
}