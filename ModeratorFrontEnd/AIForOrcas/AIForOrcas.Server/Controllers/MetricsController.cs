using AIForOrcas.DTO;
using AIForOrcas.DTO.API;
using AIForOrcas.Server.BL.Models.CosmosDB;
using AIForOrcas.Server.BL.Services;
using AIForOrcas.Server.Helpers;
using Microsoft.AspNetCore.Http;
using Microsoft.AspNetCore.Mvc;
using System;
using System.Linq;
using System.Threading.Tasks;

namespace AIForOrcas.Server.Controllers
{
	/// <summary>
	/// Endpoint for retrieving metrics information about the system and users.
	/// </summary>
	[Produces("application/json")]
	[Route("api/metrics")]
	[ApiController]
	public class MetricsController : ControllerBase
	{
		private readonly MetadataRepository _repository;

		public MetricsController(MetadataRepository repository)
		{
			_repository = repository;
		}

		private IQueryable<Metadata> BuildQueryableAsync(string timeframe, string moderator = null)
		{
			// start with all records
			var queryable = _repository.GetAll();

			// apply timeframe filter
			MetadataFilters.ApplyTimeframeFilter(ref queryable, timeframe);

			// apply moderator filter, if applicable
			MetadataFilters.ApplyModeratorFilter(ref queryable, moderator);

			return queryable;
		}

		/// <summary>
		/// Fetch system metrics.
		/// </summary>
		/// <response code="200">Returns the system's metrics.</response>
		/// <response code="204">If there are no metrics for the specified timeframe.</response>
		/// <response code="400">If the request was malformed (missing parameters).</response>
		/// <response code="500">If there is an internal error reading or processing the data from CosmosDB.</response>
		[HttpGet]
		[ProducesResponseType(typeof(Metrics), 200)]
		[ProducesResponseType(204)]
		[ProducesResponseType(400)]
		[ProducesResponseType(500)]
		[Route("system")]
		public IActionResult GetSystemMetrics([FromQuery] MetricsFilterDTO queryParameters)
		{
			try
			{
				if (string.IsNullOrWhiteSpace(queryParameters.Timeframe))
					throw new ArgumentNullException("Timeframe");

				var metrics = new Metrics();

				metrics.Timeframe = queryParameters.Timeframe;

				// Build base queryable
				var queryable = BuildQueryableAsync(queryParameters.Timeframe);

				// If not metrics to return
				if (queryable == null || queryable.Count() == 0)
				{
					return NoContent();
				}

				var results = queryable
					.Select(x => DetectionProcessors.ToDetection(x)).ToList();

				// Pull reviewed/unreviewed metrics from querable
				var reviewed = DetectionProcessors.GetReviewed(results);

				metrics.Reviewed = reviewed.ReviewedCount;
				metrics.Unreviewed = reviewed.UnreviewedCount;

				// Pull results metrics from queryable
				var detections = DetectionProcessors.GetResults(results);

				metrics.ConfirmedDetection = detections.ConfirmedCount;
				metrics.FalseDetection = detections.FalseCount;
				metrics.UnknownDetection = detections.UnknownCount;

				// Pull comments from queryable
				metrics.ConfirmedComments = DetectionProcessors.GetComments(results, "yes");
				metrics.UnconfirmedComments = DetectionProcessors.GetComments(results, "no");
				metrics.UnconfirmedComments.AddRange(DetectionProcessors.GetComments(results, "don't know"));
				metrics.UnconfirmedComments = metrics.UnconfirmedComments.OrderByDescending(x => x.Timestamp).ToList();

				// Pull tags from queryable
				metrics.Tags = DetectionProcessors.GetTags(results);

				return Ok(metrics);
			}
			catch (ArgumentNullException ex)
			{
				var details = new ProblemDetails()
				{
					Detail = ex.Message
				};
				return BadRequest(details);
			}
			catch (Exception ex)
			{
				var details = new ProblemDetails()
				{
					Title = ex.GetType().ToString(),
					Detail = ex.Message
				};

				return StatusCode(StatusCodes.Status500InternalServerError, details);
			}
		}

		/// <summary>
		/// Fetch user metrics.
		/// </summary>
		/// <response code="200">Returns the user's metrics.</response>
		/// <response code="204">If the user has no activity for the specified timeframe.</response>
		/// <response code="400">If the request was malformed (missing parameters).</response>
		/// <response code="500">If there is an internal error reading or processing the data from CosmosDB.</response>
		[HttpGet]
		[ProducesResponseType(typeof(ModeratorMetrics), 200)]
		[ProducesResponseType(204)]
		[ProducesResponseType(400)]
		[ProducesResponseType(500)]
		[Route("moderator")]
		public IActionResult GetModeratorMetrics([FromQuery] ModeratorMetricsFilterDTO queryParameters)
		{
			try
			{
				if (string.IsNullOrWhiteSpace(queryParameters.Timeframe))
					throw new ArgumentNullException("Timeframe");

				if (string.IsNullOrWhiteSpace(queryParameters.Moderator))
					throw new ArgumentNullException("Moderator");

				var metrics = new ModeratorMetrics();

				metrics.Timeframe = queryParameters.Timeframe;
				metrics.Moderator = queryParameters.Moderator;

				// Build base queryable
				var queryable = BuildQueryableAsync(queryParameters.Timeframe, queryParameters.Moderator);

				// If not metrics to return
				if (queryable == null || queryable.Count() == 0)
				{
					return NoContent();
				}

				var results = queryable
					.Select(x => DetectionProcessors.ToDetection(x)).ToList();

				//// Pull reviewed/unreviewed metrics from querable
				var reviewed = DetectionProcessors.GetReviewed(results);

				metrics.Reviewed = reviewed.ReviewedCount;
				metrics.Unreviewed = reviewed.UnreviewedCount;

				// Pull results metrics from queryable
				var detections = DetectionProcessors.GetResults(results);

				metrics.ConfirmedDetection = detections.ConfirmedCount;
				metrics.FalseDetection = detections.FalseCount;
				metrics.UnknownDetection = detections.UnknownCount;

				// Pull comments from queryable
				metrics.ConfirmedComments = DetectionProcessors.GetComments(results, "yes");
				metrics.UnconfirmedComments = DetectionProcessors.GetComments(results, "no");
				metrics.UnconfirmedComments.AddRange(DetectionProcessors.GetComments(results, "don't know"));
				metrics.UnconfirmedComments = metrics.UnconfirmedComments.OrderByDescending(x => x.Timestamp).ToList();

				// Pull tags from queryable
				metrics.Tags = DetectionProcessors.GetTags(results);

				return Ok(metrics);
			}
			catch (ArgumentNullException ex)
			{
				var details = new ProblemDetails()
				{
					Detail = ex.Message
				};
				return BadRequest(details);
			}
			catch (Exception ex)
			{
				var details = new ProblemDetails()
				{
					Title = ex.GetType().ToString(),
					Detail = ex.Message
				};

				return StatusCode(StatusCodes.Status500InternalServerError, details);
			}
		}
	}
}
