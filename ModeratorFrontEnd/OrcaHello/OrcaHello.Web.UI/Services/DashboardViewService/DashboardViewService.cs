using Azure.Core;
using Azure;
using OrcaHello.Web.Shared.Models.Detections;
using OrcaHello.Web.UI.Models;
using static System.Runtime.InteropServices.JavaScript.JSType;
using System.Collections.Generic;
using System.Drawing;
using System;

namespace OrcaHello.Web.UI.Services
{
    // DashboardViewService implements the presentation layer of the software application,
    // according to the Standard. It is responsible for retrieving and displaying summary and metrics for the system and
    // specific moderator, using the DetectionService, TagService, MetricsService, CommentService, and ModeratorService as 
    // dependencies and the LoggingUtilities as a helper.
    public partial class DashboardViewService : IDashboardViewService
    {
        private readonly IDetectionService _detectionService;
        private readonly ITagService _tagService;
        private readonly IMetricsService _metricsService;
        private readonly ICommentService _commentService;
        private readonly IModeratorService _moderatorService;
        private readonly ILogger<DashboardViewService> _logger;

        public DashboardViewService(IDetectionService detectionService,
            ITagService tagService,
            IMetricsService metricsService,
            ICommentService commentService,
            IModeratorService moderatorService,
            ILogger<DashboardViewService> logger)
        {
            _detectionService = detectionService;
            _tagService = tagService;
            _metricsService = metricsService;
            _commentService = commentService;
            _moderatorService = moderatorService;
            _logger = logger;
        }

        #region system metrics

        //Retrieves a list of tags from the tag service, filtered by the date range specified in the request. It calls the broker service,
        //validates the response, and handles any exceptions.
        public ValueTask<List<string>> RetrieveFilteredTagsAsync(TagsByDateRequest request) =>
        TryCatch(async () =>
        {
            ValidateRequest(request);
            ValidateDateRange(request.FromDate, request.ToDate);

            TagListForTimeframeResponse response =
                await _tagService.RetrieveFilteredTagsAsync(
                    fromDate: request.FromDate,
                    toDate: request.ToDate);

            ValidateResponse(response);

            return response.Tags;
        });

        //Retrieves a metrics object from the MetricsService, filtered by the date range specified in the request. It calls the broker service,
        //validates the response, and maps it to a metrics item view response.
        public ValueTask<MetricsItemViewResponse> RetrieveFilteredMetricsAsync(MetricsByDateRequest request) =>
        TryCatch(async () =>
        {
            ValidateRequest(request);
            ValidateDateRange(request.FromDate, request.ToDate);

            MetricsResponse response =
                await _metricsService.RetrieveFilteredMetricsAsync(
            fromDate: request.FromDate,
            toDate: request.ToDate);

            ValidateResponse(response);

            return MetricsItemView.AsMetricsItemViewResponse(response);
        });
        
        // Retrieves a list of detections for a given tag from the DetectionService, filtered by the date range
        // and paginated by the page number and size specified in the request. It calls the broker service,
        // validates the response, and maps it to a detection item view response.
        public ValueTask<DetectionItemViewResponse> RetrieveFilteredDetectionsForTagsAsync(PaginatedDetectionsByTagAndDateRequest request) =>
        TryCatch(async () =>
        {
            ValidateRequest(request);
            ValidateTag(request.Tag);
            ValidateDateRange(request.FromDate, request.ToDate);
            ValidatePagination(request.Page, request.PageSize);

            DetectionListForTagResponse response =
            await _detectionService.RetrieveFilteredAndPaginatedDetectionsForTagAsync(
            tag: request.Tag,
            fromDate: request.FromDate,
            toDate: request.ToDate,
            page: request.Page,
            pageSize: request.PageSize);

            ValidateResponse(response);

            return new DetectionItemViewResponse
            {
                DetectionItemViews = response.Detections.Select(DetectionItemView.AsDetectionItemView).ToList(),
                Count = response.TotalCount
            };
        });
        
        // Retrieves a list of positive comments from the CommentService, filtered by the date range and paginated
        // by the page number and size specified in the request.It calls the broker service, validates the response,
        // and maps it to a comment item view response.
        public ValueTask<CommentItemViewResponse> RetrieveFilteredPositiveCommentsAsync(PaginatedCommentsByDateRequest request) =>
        TryCatch(async () =>
        {
            ValidateRequest(request);
            ValidateDateRange(request.FromDate, request.ToDate);
            ValidatePagination(request.Page, request.PageSize);

            CommentListResponse response =
            await _commentService.RetrieveFilteredPositiveCommentsAsync(
            fromDate: request.FromDate,
            toDate: request.ToDate,
            page: request.Page,
            pageSize: request.PageSize);

            ValidateResponse(response);

            return new CommentItemViewResponse
            {
                CommentItemViews = response.Comments.Select(CommentItemView.AsCommentItemView).ToList(),
                Count = response.TotalCount
            };
        });

        // Retrieves a list of negative and unknown comments from the CommentService, filtered by the date range and paginated
        // by the page number and size specified in the request. It calls the broker service, validates the response,
        // and maps it to a comment item view response.
        public ValueTask<CommentItemViewResponse> RetrieveFilteredNegativeAndUnknownCommentsAsync(PaginatedCommentsByDateRequest request) =>
        TryCatch(async () =>
        {
            ValidateRequest(request);
            ValidateDateRange(request.FromDate, request.ToDate);
            ValidatePagination(request.Page, request.PageSize);

            CommentListResponse response =
                await _commentService.RetrieveFilteredNegativeAndUnknownCommentsAsync(
                    fromDate: request.FromDate,
                    toDate: request.ToDate,
                    page: request.Page,
                    pageSize: request.PageSize);

            ValidateResponse(response);

            return new CommentItemViewResponse
            {
                CommentItemViews = response.Comments.Select(CommentItemView.AsCommentItemView).ToList(),
                Count = response.TotalCount
            };
        });

        #endregion

        #region moderator metrics

        // Retrieves a list of tags for a given moderator from the ModeratorService, filtered by the date range specified in the request.
        // It calls the broker service, validates the response, and handles any exceptions.
        public ValueTask<List<string>> RetrieveFilteredTagsForModeratorAsync(string moderator, TagsByDateRequest request) =>
        TryCatch(async () =>
        {
            ValidateRequest(request);
            ValidateModerator(moderator);
            ValidateDateRange(request.FromDate, request.ToDate);

            TagListForModeratorResponse response =
                await _moderatorService.GetFilteredTagsForModeratorAsync(
                    moderator: moderator,
                    fromDate: request.FromDate,
                    toDate: request.ToDate);

            ValidateResponse(response);

            return response.Tags;
        });

        // Retrieves a metrics object for a given moderator from the ModeratorService, filtered by the date range specified in
        // the request. It calls the broker service, validates the response, and maps it to a moderator metrics item view response.
        public ValueTask<ModeratorMetricsItemViewResponse> RetrieveFilteredMetricsForModeratorAsync(string moderator, MetricsByDateRequest request) =>
        TryCatch(async () =>
        {
            ValidateRequest(request);
            ValidateModerator(moderator);
            ValidateDateRange(request.FromDate, request.ToDate);

            MetricsForModeratorResponse response =
                await _moderatorService.GetFilteredMetricsForModeratorAsync(
                    moderator: moderator,
                    fromDate: request.FromDate,
                    toDate: request.ToDate);

            ValidateResponse(response);

            return AsModeratorMetricsItemViewResponse(response);
        });

        // Retrieves a list of positive comments for a given moderator from the ModeratorService, filtered by the date range and
        // paginated by the page number and size specified in the request. It calls the broker service, validates the response, and maps it to a
        // moderator comment item view response.
        public ValueTask<ModeratorCommentItemViewResponse> RetrieveFilteredPositiveCommentsForModeratorAsync(string moderator, PaginatedCommentsByDateRequest request) =>
        TryCatch(async () =>
        {
            ValidateRequest(request);
            ValidateModerator(moderator);
            ValidateDateRange(request.FromDate, request.ToDate);
            ValidatePagination(request.Page, request.PageSize);

            CommentListForModeratorResponse response =
                await _moderatorService.GetFilteredPositiveCommentsForModeratorAsync(
                    moderator: moderator,
                    fromDate: request.FromDate,
                    toDate: request.ToDate,
                    page: request.Page,
                    pageSize: request.PageSize);

            ValidateResponse(response);

            return new ModeratorCommentItemViewResponse
            {
                CommentItemViews = response.Comments.Select(CommentItemView.AsCommentItemView).ToList(),
                Count = response.TotalCount,
                Moderator = moderator
            };
        });

        // Retrieves a list of negative and unknown comments for a given moderator from the ModeratorService, filtered by the date range
        // and paginated by the page number and size specified in the request. It calls the broker service, validates the response,
        // and maps it to a moderator comment item view response.
        public ValueTask<ModeratorCommentItemViewResponse> RetrieveFilteredNegativeAndUnknownCommentsForModeratorAsync(string moderator, PaginatedCommentsByDateRequest request) =>
        TryCatch(async () =>
        {
            ValidateRequest(request);
            ValidateModerator(moderator);
            ValidateDateRange(request.FromDate, request.ToDate);
            ValidatePagination(request.Page, request.PageSize);

            CommentListForModeratorResponse response =
                await _moderatorService.GetFilteredNegativeAndUknownCommentsForModeratorAsync(
                    moderator: moderator,
                    fromDate: request.FromDate,
                    toDate: request.ToDate,
                    page: request.Page,
                    pageSize: request.PageSize);

            ValidateResponse(response);

            return new ModeratorCommentItemViewResponse
            {
                CommentItemViews = response.Comments.Select(CommentItemView.AsCommentItemView).ToList(),
                Count = response.TotalCount,
                Moderator = moderator
            };
        });
        // Retrieves a list of detections for a given tag and moderator from the ModeratorService, filtered by the date range and
        // paginated by the page number and size specified in the request. It calls the broker service, validates the response,
        // and maps it to a moderator detection item view response.
        public ValueTask<ModeratorDetectionItemViewResponse> RetrieveFilteredDetectionsForTagAndModeratorAsync(string moderator, PaginatedDetectionsByTagAndDateRequest request) =>
        TryCatch(async () =>
        {
            ValidateRequest(request);
            ValidateModerator(moderator);
            ValidateTag(request.Tag);
            ValidateDateRange(request.FromDate, request.ToDate);
            ValidatePagination(request.Page, request.PageSize);

            DetectionListForModeratorAndTagResponse response =
                await _moderatorService.GetFilteredDetectionsForTagAndModeratorAsync(
                    moderator: moderator,
                    tag: request.Tag,
                    fromDate: request.FromDate,
                    toDate: request.ToDate,
                    page: request.Page,
                    pageSize: request.PageSize);

            ValidateResponse(response);

            return new ModeratorDetectionItemViewResponse
            {
                DetectionItemViews = response.Detections
                .Select(DetectionItemView.AsDetectionItemView).ToList(),
                Count = response.TotalCount
            };
        });

        private static Func<MetricsForModeratorResponse, ModeratorMetricsItemViewResponse> AsModeratorMetricsItemViewResponse =>
        metricsResponse => new ModeratorMetricsItemViewResponse
        {
            MetricsItemViews = new List<MetricsItemView>()
            {
                new(DetectionState.Positive.ToString(), metricsResponse.Positive, "#468f57"),
                new(DetectionState.Negative.ToString(), metricsResponse.Negative, "#bb595f"),
                new(DetectionState.Unknown.ToString(), metricsResponse.Unknown, "#bc913e")
            },
            FromDate = metricsResponse.FromDate,
            ToDate = metricsResponse.ToDate,
            Moderator = metricsResponse.Moderator,
            };

        #endregion
    }
}
