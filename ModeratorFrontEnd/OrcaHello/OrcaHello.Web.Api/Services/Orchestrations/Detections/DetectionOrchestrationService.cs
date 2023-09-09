﻿namespace OrcaHello.Web.Api.Services
{
    public partial class DetectionOrchestrationService : IDetectionOrchestrationService
    {
        private readonly IMetadataService _metadataService;
        private readonly ILogger<DetectionOrchestrationService> _logger;

        // Needed for unit testing wrapper to work properly

        public DetectionOrchestrationService() { }

        public DetectionOrchestrationService(IMetadataService metadataService,
            ILogger<DetectionOrchestrationService> logger)
        {
            _metadataService = metadataService;
            _logger = logger;
        }

        public ValueTask<DetectionListForTagResponse> RetrieveDetectionsForGivenTimeframeAndTagAsync(DateTime? fromDate, DateTime? toDate, string tag, int page, int pageSize) =>
        TryCatch(async () =>
        {
            Validate(fromDate, nameof(fromDate));
            Validate(toDate, nameof(toDate));
            Validate(tag, nameof(tag));
            ValidatePage(page);
            ValidatePageSize(pageSize);

            DateTime nonNullableFromDate = fromDate ?? default;
            DateTime nonNullableToDate = toDate ?? default;

            QueryableMetadataForTimeframeAndTag results = await _metadataService.
                RetrieveMetadataForGivenTimeframeAndTagAsync(nonNullableFromDate, nonNullableToDate, tag, page, pageSize);

            return new DetectionListForTagResponse
            {
                Detections = results.QueryableRecords.Select(r => AsDetection(r)).ToList(),
                FromDate = results.FromDate,
                ToDate = results.ToDate,
                Tag = results.Tag,
                Page = results.Page,
                PageSize = results.PageSize,
                TotalCount = results.TotalCount,
                Count = results.QueryableRecords.Count()
            };
        });

        public ValueTask<DetectionListResponse> RetrieveFilteredDetectionsAsync(DateTime? fromDate, DateTime? toDate, string state, string sortBy, bool isDescending, 
            string location, int page, int pageSize) =>
        TryCatch(async () =>
        {
            Validate(fromDate, nameof(fromDate));
            Validate(toDate, nameof(toDate));
            Validate(state, nameof(state));
            Validate(sortBy, nameof(sortBy));
            ValidatePage(page);
            ValidatePageSize(pageSize);

            ValidateStateIsAcceptable(state);

            DateTime nonNullableFromDate = fromDate ?? default;
            DateTime nonNullableToDate = toDate ?? default;

            QueryableMetadataFiltered results = await _metadataService.
                RetrievePaginatedMetadataAsync(state, nonNullableFromDate, nonNullableToDate, sortBy, isDescending, location, page, pageSize);

            return new DetectionListResponse
            {
                Detections = results.QueryableRecords.Select(r => AsDetection(r)).ToList(),
                FromDate = results.FromDate,
                ToDate = results.ToDate,
                Page = results.Page,
                PageSize = results.PageSize,
                TotalCount = results.TotalCount,
                Count = results.QueryableRecords.Count(),
                State = results.State,
                SortBy = results.SortBy,
                SortOrder = results.SortOrder,
                Location = results.Location
            };
        });

        public ValueTask<Detection> RetrieveDetectionByIdAsync(string id) =>
        TryCatch(async () =>
        {
            Validate(id, nameof(id));
            Metadata maybeMetadata = await _metadataService.RetrieveMetadataByIdAsync(id);
            ValidateStorageMetadata(maybeMetadata, id);

            return AsDetection(maybeMetadata);
        });

        public ValueTask<Detection> ModerateDetectionByIdAsync(string id, ModerateDetectionRequest request) =>
        TryCatch(async () =>
        {
            Validate(id, nameof(id));
            ValidateModerateRequestOnUpdate(request);

            // Get the current record
            Metadata existingRecord = await _metadataService.RetrieveMetadataByIdAsync(id);
            ValidateStorageMetadata(existingRecord, id);

            var existingState = existingRecord.State;

            // Make updates so they can be added as a new record
            Metadata newRecord = existingRecord;
            newRecord.State = request.State;
            newRecord.Moderator = request.Moderator;
            newRecord.DateModerated = request.DateModerated.ToString();
            newRecord.Comments = request.Comments;
            newRecord.Tags = request.Tags;

            bool existingRecordDeleted = await _metadataService.RemoveMetadataByIdAndStateAsync(id, existingState);
            ValidateDeleted(existingRecordDeleted, id);

            bool newRecordCreated = await _metadataService.AddMetadataAsync(newRecord);

            if(!newRecordCreated)
            {
                bool existingRecordRecreated = await _metadataService.AddMetadataAsync(existingRecord);
                ValidateInserted(existingRecordRecreated, id);
            }

            return AsDetection(newRecord);
        });

        public ValueTask<DetectionListForInterestLabelResponse> RetrieveDetectionsForGivenInterestLabelAsync(string interestLabel) =>
         TryCatch(async () =>
         {
             Validate(interestLabel, nameof(interestLabel));

             QueryableMetadata results = await _metadataService.
                 RetrieveMetadataForInterestLabelAsync(interestLabel);

             return new DetectionListForInterestLabelResponse
             {
                 Detections = results.QueryableRecords.Select(r => AsDetection(r)).ToList(),
                 InterestLabel = interestLabel,
                 TotalCount = results.QueryableRecords.Count()
             };
         });

        private static Detection AsDetection(Metadata metadata)
        {
            var detection = new Detection
            {
                Id = metadata.Id,
                AudioUri = metadata.AudioUri,
                SpectrogramUri = metadata.ImageUri,
                State = metadata.State,
                LocationName = metadata.LocationName,
                Timestamp = metadata.Timestamp,
                Tags = metadata.Tags,
                Comments = metadata.Comments,
                Confidence = metadata.WhaleFoundConfidence,
                Moderator = metadata.Moderator,
                Moderated = !string.IsNullOrWhiteSpace(metadata.DateModerated) ? DateTime.Parse(metadata.DateModerated) : null,
            };

            if(metadata.Location != null)
            {
                detection.Location = new Shared.Models.Detections.Location
                {
                    Name = metadata.Location.Name,
                    Longitude = metadata.Location.Longitude,
                    Latitude = metadata.Location.Latitude
                };
            }

            detection.Annotations = metadata.Predictions.Select(r => AsAnnotation(r)).ToList();

            return detection;
        }

        private static Annotation AsAnnotation(Prediction prediction)
        {
            return new Annotation
            {
                Id = prediction.Id,
                StartTime = prediction.StartTime,
                EndTime = prediction.StartTime + prediction.Duration,
                Confidence = prediction.Confidence
            };
        }
    }
}
