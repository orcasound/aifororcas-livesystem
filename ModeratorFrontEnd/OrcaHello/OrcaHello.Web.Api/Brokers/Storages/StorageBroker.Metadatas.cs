using Location = OrcaHello.Web.Api.Models.Location;

namespace OrcaHello.Web.Api.Brokers.Storages
{
    public partial class StorageBroker
    {
        #region Tag Queries

        public async Task<List<string>> GetAllTagList()
        {
            var queryDefinition = new QueryDefinition("SELECT DISTINCT VALUE tag FROM c JOIN tag IN c.tags");

            return await ExecuteListStringQuery(queryDefinition);
        }

        public async Task<List<string>> GetTagListByTimeframe(DateTime fromDate, DateTime toDate)
        {
            var queryDefinition = new QueryDefinition("SELECT DISTINCT VALUE tag FROM c JOIN tag IN c.tags WHERE c.timestamp BETWEEN @startTime AND @endTime")
                .WithParameter("@startTime", CosmosUtilities.FormatDate(fromDate))
                .WithParameter("@endTime", CosmosUtilities.FormatDate(toDate));

            return await ExecuteListStringQuery(queryDefinition);
        }

        public async Task<List<string>> GetTagListByTimeframeAndModerator(DateTime fromDate, DateTime toDate, string moderator)
        {
            var queryDefinition = new QueryDefinition("SELECT DISTINCT VALUE tag FROM c JOIN tag IN c.tags WHERE (c.timestamp BETWEEN @startTime AND @endTime) AND c.moderator = @moderator")
                .WithParameter("@startTime", CosmosUtilities.FormatDate(fromDate))
                .WithParameter("@endTime", CosmosUtilities.FormatDate(toDate))
                .WithParameter("@moderator", moderator);

            return await ExecuteListStringQuery(queryDefinition);
        }

        private async Task<List<string>> ExecuteListStringQuery(QueryDefinition queryDefinition)
        {
            var queryIterator = _detectionsContainer.GetItemQueryIterator<string>(queryDefinition);
            var results = new List<string>();

            while (queryIterator.HasMoreResults)
            {
                var response = await queryIterator.ReadNextAsync();
                results.AddRange(response);
            }

            return results;
        }

        #endregion

        #region Interest Label Queries

        public async Task<List<string>> GetAllInterestLabels()
        {
            var queryDefinition = new QueryDefinition("SELECT DISTINCT VALUE c.interestLabel FROM c WHERE IS_DEFINED(c.interestLabel)");

            return await ExecuteListStringQuery(queryDefinition);
        }

        #endregion

        #region Moderator Queries

        public async Task<List<string>> GetModeratorList()
        {
            var queryDefinition = new QueryDefinition("SELECT DISTINCT VALUE c.moderator FROM c WHERE IS_DEFINED(c.moderator) AND c.moderator != null");

            return await ExecuteListStringQuery(queryDefinition);
        }

        #endregion

        #region Metadata Queries

        public async Task<Metadata> GetMetadataById(string id)
        {
            var queryDefinition = new QueryDefinition("SELECT * FROM c WHERE c.id = @id")
                .WithParameter("@id", id);

            Metadata result = null!;

            var queryIterator = _detectionsContainer.GetItemQueryIterator<Metadata>(queryDefinition);

            while (queryIterator.HasMoreResults)
            {
                var response = await queryIterator.ReadNextAsync();

                if (response.Count > 0)
                {
                    result = response.First();
                }
            }

            return result;
        }

        public async Task<bool> DeleteMetadataByIdAndState(string id, string currentState)
        {
            ItemResponse<Metadata> deleteResponse = await _detectionsContainer.DeleteItemAsync<Metadata>(id, new PartitionKey(currentState));

            return deleteResponse.StatusCode == System.Net.HttpStatusCode.NoContent;
        }

        public async Task<bool> InsertMetadata(Metadata metadata)
        {
            ItemResponse<Metadata> insertResponse = await _detectionsContainer.CreateItemAsync(metadata, new PartitionKey(metadata.State));

            return insertResponse.StatusCode == System.Net.HttpStatusCode.Created;
        }

        public async Task<bool> UpdateMetadataInPartition(Metadata metadata)
        {
            var updateResponse = await _detectionsContainer.ReplaceItemAsync(metadata, metadata.Id, new PartitionKey(metadata.State));

            return updateResponse.StatusCode == System.Net.HttpStatusCode.OK;
        }

        public async Task<ListMetadataAndCount> GetMetadataListByTimeframeAndTag(DateTime fromDate, DateTime toDate,
            List<string> tags, string tagOperator, int page = 1, int pageSize = 10)
        {
            string queryText = "SELECT * FROM c WHERE(c.timestamp BETWEEN @startTime AND @endTime) AND (";
            for (int i = 0; i < tags.Count; i++)
            {
                if (i > 0)
                {
                    queryText += $"{tagOperator} ";
                }
                queryText += $"ARRAY_CONTAINS(c.tags, @tag{i}) ";
            }

            queryText += ") ORDER BY c.timestamp ASC";

            var queryDefinition = new QueryDefinition(queryText)
                .WithParameter("@startTime", CosmosUtilities.FormatDate(fromDate))
                .WithParameter("@endTime", CosmosUtilities.FormatDate(toDate));

            for (int i = 0; i < tags.Count; i++)
            {
                queryDefinition.WithParameter($"@tag{i}", tags[i]);
            }

            return await ExecutePaginatedMetadataQuery(queryDefinition, page, pageSize);
        }

        public async Task<ListMetadataAndCount> GetMetadataListByTimeframeTagAndModerator(DateTime fromDate, DateTime toDate,
        string moderator, string tag, int page = 1, int pageSize = 10)
        {
            var queryDefinition = new QueryDefinition($"SELECT * FROM c WHERE (c.timestamp BETWEEN @startTime AND @endTime) AND c.moderator = @moderator AND ARRAY_CONTAINS(c.tags, @tag) ORDER BY c.timestamp ASC")
                .WithParameter("@startTime", CosmosUtilities.FormatDate(fromDate))
                .WithParameter("@endTime", CosmosUtilities.FormatDate(toDate))
                .WithParameter("@moderator", moderator)
                .WithParameter("@tag", tag);

            return await ExecutePaginatedMetadataQuery(queryDefinition, page, pageSize);
        }

        public async Task<ListMetadataAndCount> GetAllMetadataListByTag(string tag)
        {
            var queryDefinition = new QueryDefinition("SELECT * FROM c WHERE ARRAY_CONTAINS(c.tags, @tag) ORDER BY c.timestamp ASC")
                .WithParameter("@tag", tag);

            return await ExecuteMetadataQuery(queryDefinition);
        }

        public async Task<ListMetadataAndCount> GetAllMetadataListByInterestLabel(string interestLabel)
        {
            var queryDefinition = new QueryDefinition("SELECT * FROM c WHERE c.interestLabel = @interestLabel")
                .WithParameter("@interestLabel", interestLabel);

            return await ExecuteMetadataQuery(queryDefinition);
        }

        public async Task<ListMetadataAndCount> GetPositiveMetadataListByTimeframe(DateTime fromDate, DateTime toDate,
            int page = 1, int pageSize = 10)
        {
            var queryDefinition = new QueryDefinition($"SELECT * FROM c WHERE c.state = '{DetectionState.Positive}' AND (c.timestamp BETWEEN @startTime AND @endTime) ORDER BY c.timestamp ASC")
                .WithParameter("@startTime", CosmosUtilities.FormatDate(fromDate))
                .WithParameter("@endTime", CosmosUtilities.FormatDate(toDate));

            return await ExecutePaginatedMetadataQuery(queryDefinition, page, pageSize);
        }

        public async Task<ListMetadataAndCount> GetPositiveMetadataListByTimeframeAndModerator(DateTime fromDate, DateTime toDate,
            string moderator, int page = 1, int pageSize = 10)
        {
            var queryDefinition = new QueryDefinition($"SELECT * FROM c WHERE c.state = '{DetectionState.Positive}' AND (c.timestamp BETWEEN @startTime AND @endTime) AND c.moderator = @moderator ORDER BY c.timestamp ASC")
                .WithParameter("@startTime", CosmosUtilities.FormatDate(fromDate))
                .WithParameter("@endTime", CosmosUtilities.FormatDate(toDate))
                .WithParameter("@moderator", moderator);

            return await ExecutePaginatedMetadataQuery(queryDefinition, page, pageSize);
        }

        public async Task<ListMetadataAndCount> GetNegativeAndUnknownMetadataListByTimeframe(DateTime fromDate, DateTime toDate,
            int page = 1, int pageSize = 10)
        {
            var queryDefinition = new QueryDefinition($"SELECT * FROM c WHERE c.state IN ( '{DetectionState.Negative}', '{DetectionState.Unknown}') AND (c.timestamp BETWEEN @startTime AND @endTime) ORDER BY c.timestamp ASC")
                .WithParameter("@startTime", CosmosUtilities.FormatDate(fromDate))
                .WithParameter("@endTime", CosmosUtilities.FormatDate(toDate));

            return await ExecutePaginatedMetadataQuery(queryDefinition, page, pageSize);
        }

        public async Task<ListMetadataAndCount> GetNegativeAndUnknownMetadataListByTimeframeAndModerator(DateTime fromDate, DateTime toDate,
            string moderator, int page = 1, int pageSize = 10)
        {
            var queryDefinition = new QueryDefinition($"SELECT * FROM c WHERE c.state IN ('{DetectionState.Negative}', '{DetectionState.Unknown}') AND (c.timestamp BETWEEN @startTime AND @endTime) AND c.moderator = @moderator ORDER BY c.timestamp ASC")
                .WithParameter("@startTime", CosmosUtilities.FormatDate(fromDate))
                .WithParameter("@endTime", CosmosUtilities.FormatDate(toDate))
                .WithParameter("@moderator", moderator);

            return await ExecutePaginatedMetadataQuery(queryDefinition, page, pageSize);
        }

        public async Task<ListMetadataAndCount> GetUnreviewedMetadataListByTimeframe(DateTime fromDate, DateTime toDate,
            int page = 1, int pageSize = 10)
        {
            var queryDefinition = new QueryDefinition($"SELECT * FROM c WHERE c.state = '{DetectionState.Unreviewed}' AND (c.timestamp BETWEEN @startTime AND @endTime) ORDER BY c.timestamp ASC")
                .WithParameter("@startTime", CosmosUtilities.FormatDate(fromDate))
                .WithParameter("@endTime", CosmosUtilities.FormatDate(toDate));

            return await ExecutePaginatedMetadataQuery(queryDefinition, page, pageSize);
        }

        public async Task<ListMetadataAndCount> GetMetadataListFiltered(string state, DateTime fromDate, DateTime toDate, string sortBy,
            string sortOrder, string location, int page = 1, int pageSize = 1)
        {
            var queryText = "SELECT * " +
                "FROM c " +
                "WHERE c.state = @state " +
                "AND (c.timestamp BETWEEN @startTime AND @endTime) ";

            if (!string.IsNullOrEmpty(location))
            {
                queryText += "AND (c.locationName = @location OR c.locationName = '') ";
            }

            queryText += $"ORDER BY c.{sortBy} {sortOrder}";

            var queryDefinition = new QueryDefinition(queryText)
                .WithParameter("@state", state)
                .WithParameter("@startTime", CosmosUtilities.FormatDate(fromDate))
                .WithParameter("@endTime", CosmosUtilities.FormatDate(toDate))
                .WithParameter("@location", location);

            return await ExecutePaginatedMetadataQuery(queryDefinition, page, pageSize);
        }

        private async Task<ListMetadataAndCount> ExecuteMetadataQuery(QueryDefinition queryDefinition)
        {
            var queryIterator = _detectionsContainer.GetItemQueryIterator<Metadata>(queryDefinition);
            var fullResults = new List<Metadata>();

            while (queryIterator.HasMoreResults)
            {
                var response = await queryIterator.ReadNextAsync();
                fullResults.AddRange(response);
            }
            ListMetadataAndCount results = new()
            {
                PaginatedRecords = fullResults,
                TotalCount = fullResults.Count
            };

            return results;
        }

        private async Task<ListMetadataAndCount> ExecutePaginatedMetadataQuery(QueryDefinition queryDefinition, int page, int pageSize)
        {
            var queryIterator = _detectionsContainer.GetItemQueryIterator<Metadata>(queryDefinition);
            var fullResults = new List<Metadata>();

            while (queryIterator.HasMoreResults)
            {
                var response = await queryIterator.ReadNextAsync();
                fullResults.AddRange(response);
            }

            ListMetadataAndCount results = new()
            {
                PaginatedRecords = fullResults.Skip((page - 1) * pageSize).Take(pageSize).ToList(),
                TotalCount = fullResults.Count
            };

            return results;
        }

        #endregion

        #region Metrics Queries

        public async Task<List<MetricResult>> GetMetricsListByTimeframe(DateTime fromDate, DateTime toDate)
        {
            var queryDefinition = new QueryDefinition("SELECT c.state AS State, COUNT(1) AS Count FROM c WHERE (c.timestamp BETWEEN @startTime AND @endTime) GROUP BY c.state")
                .WithParameter("@startTime", CosmosUtilities.FormatDate(fromDate))
                .WithParameter("@endTime", CosmosUtilities.FormatDate(toDate));

            return await ExecuteMetricsQuery(queryDefinition);
        }

        public async Task<List<MetricResult>> GetMetricsListByTimeframeAndModerator(DateTime fromDate, DateTime toDate, string moderator)
        {
            var queryDefinition = new QueryDefinition("SELECT c.state AS State, COUNT(1) AS Count FROM c WHERE (c.timestamp BETWEEN @startTime AND @endTime) AND c.moderator = @moderator GROUP BY c.state")
                .WithParameter("@startTime", CosmosUtilities.FormatDate(fromDate))
                .WithParameter("@endTime", CosmosUtilities.FormatDate(toDate))
                .WithParameter("@moderator", moderator);

            return await ExecuteMetricsQuery(queryDefinition);
        }

        private async Task<List<MetricResult>> ExecuteMetricsQuery(QueryDefinition queryDefinition)
        {
            var queryIterator = _detectionsContainer.GetItemQueryIterator<MetricResult>(queryDefinition);
            var results = new List<MetricResult>();

            while (queryIterator.HasMoreResults)
            {
                var response = await queryIterator.ReadNextAsync();
                results.AddRange(response);
            }

            return results;
        }

        #endregion
    }
}
