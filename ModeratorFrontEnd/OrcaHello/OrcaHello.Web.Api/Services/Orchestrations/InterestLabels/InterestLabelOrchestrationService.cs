namespace OrcaHello.Web.Api.Services
{
    public partial class InterestLabelOrchestrationService : IInterestLabelOrchestrationService
    {
        private readonly IMetadataService _metadataService;
        private readonly ILogger<InterestLabelOrchestrationService> _logger;

        // Needed for unit testing wrapper to work properly

        public InterestLabelOrchestrationService() { }

        public InterestLabelOrchestrationService(IMetadataService metadataService,
            ILogger<InterestLabelOrchestrationService> logger)
        {
            _metadataService = metadataService;
            _logger = logger;
        }

        public ValueTask<InterestLabelListResponse> RetrieveAllInterestLabelsAsync() =>
        TryCatch(async () =>
        {
            var candidateRecords = await _metadataService.RetrieveAllInterestLabelsAsync();

            return new InterestLabelListResponse
            {
                InterestLabels = candidateRecords.QueryableRecords.OrderBy(s => s).ToList(),
                Count = candidateRecords.TotalCount
            };
        });

        public ValueTask<InterestLabelRemovalResponse> RemoveInterestLabelFromDetectionAsync(string id) =>
        TryCatch(async () =>
         {
             Validate(id, nameof(id));

             Metadata existingRecord = await _metadataService.RetrieveMetadataByIdAsync(id);
             ValidateMetadataFound(existingRecord, id);

             InterestLabelRemovalResponse result = new()
             {
                 LabelRemoved = existingRecord.InterestLabel,
                 Id = id
             };

             // Make updates so they can be added as a new record
             Metadata newRecord = existingRecord;
             newRecord.InterestLabel = null!;

             bool existingRecordDeleted = await _metadataService.RemoveMetadataByIdAndStateAsync(id, existingRecord.State);
             ValidateDeleted(existingRecordDeleted, id);

             bool newRecordCreated = await _metadataService.AddMetadataAsync(newRecord);

             if (!newRecordCreated)
             {
                 bool existingRecordRecreated = await _metadataService.AddMetadataAsync(existingRecord);
                 ValidateInserted(existingRecordRecreated, id);
             }

             return result;
         });

        public ValueTask<InterestLabelAddResponse> AddInterestLabelToDetectionAsync(string id, string interestLabel) =>
        TryCatch(async () =>
        {
            Validate(id, nameof(id));
            Validate(interestLabel, nameof(interestLabel));

            Metadata storedMetadata = await _metadataService.RetrieveMetadataByIdAsync(id);
            ValidateMetadataFound(storedMetadata, id);

            storedMetadata.InterestLabel = interestLabel;

            var updatedMetdata = await _metadataService.UpdateMetadataAsync(storedMetadata);

            InterestLabelAddResponse result = new()
            {
                LabelAdded = storedMetadata.InterestLabel,
                Id = id
            };

            return result;
        });
    }
}
