using AIForOrcas.DTO.API;
using AIForOrcas.Server.BL.Services;
using AIForOrcas.Server.Helpers;
using Microsoft.AspNetCore.Http;
using Microsoft.AspNetCore.Mvc;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;

namespace AIForOrcas.Server.Controllers
{
	/// <summary>
	/// Endpoint for retrieving and updating detection items.
	/// </summary>
	[Produces("application/json")]
	[Route("api/detections")]
	[ApiController]
	public class DetectionsController : ControllerBase
	{
		private readonly MetadataRepository _repository;

		public DetectionsController(MetadataRepository repository)
		{
			_repository = repository;
		}

		#region Helpers

		private void SetHeaderCounts(double totalRecords, int recordsPerPage)
		{
			double totalAmountPages = Math.Ceiling(totalRecords / recordsPerPage);

			HttpContext.Response.Headers.Add("totalNumberRecords", totalRecords.ToString());
			HttpContext.Response.Headers.Add("totalAmountPages", totalAmountPages.ToString());
		}

		#endregion

		/// <summary>
		/// List all AI/ML generated detections, regardless of review status.
		/// </summary>
		/// <response code="200">Returns the list of all detections.</response>
		/// <response code="204">If there are no detections for the specified timeframe.</response>
		/// <response code="400">If the request was malformed (missing parameters).</response>
		/// <response code="500">If there is an internal error reading or processing the data from CosmosDB.</response>
		[HttpGet]
		[ProducesResponseType(typeof(IEnumerable<Detection>), 200)]
		[ProducesResponseType(204)]
		[ProducesResponseType(400)]
		[ProducesResponseType(500)]
		public IActionResult Get([FromQuery] DetectionQueryParameters queryParameters)
		{
			try
			{
				if (string.IsNullOrWhiteSpace(queryParameters.Timeframe))
					throw new ArgumentNullException("Timeframe");

				if (queryParameters.DateFrom > queryParameters.DateTo)
					throw new Exception("From Date should be less than To date");

				if (string.IsNullOrWhiteSpace(queryParameters.SortBy))
					throw new ArgumentNullException("SortBy");

				if (string.IsNullOrWhiteSpace(queryParameters.SortOrder))
					throw new ArgumentNullException("SortOrder");

				if (string.IsNullOrWhiteSpace(queryParameters.Location))
					throw new ArgumentNullException("Location");

				if (queryParameters.Page == 0)
					throw new ArgumentNullException("Page");

				if (queryParameters.RecordsPerPage == 0)
					throw new ArgumentNullException("RecordsPerPage");

				// start with all records
				var queryable = _repository.GetAll();

				// apply timeframe filter
				MetadataFilters.ApplyTimeframeFilter(ref queryable, queryParameters.Timeframe, queryParameters.DateFrom,queryParameters.DateTo);

				// apply location filter
				if(queryParameters.Location.ToLower() != "all")
				{
					MetadataFilters.ApplyLocationFilter(ref queryable, queryParameters.Location);
				}

				// If no detections found
				if (queryable == null || queryable.Count() == 0)
				{
					return NoContent();
				}

				// total number of records
				double recordCount = queryable.Count();

				var results = queryable
					.Select(x => DetectionProcessors.ToDetection(x)).ToList();

				// apply sort filter
				if (queryParameters.SortBy.ToLower() == "confidence")
					DetectionFilters.ApplyConfidenceSortFilter(ref results, queryParameters.SortOrder);
				else if (queryParameters.SortBy.ToLower() == "timestamp")
					DetectionFilters.ApplyTimestampSortFilter(ref results, queryParameters.SortOrder);

				// apply pagination filter
				DetectionFilters.ApplyPaginationFilter(ref results, queryParameters.Page, queryParameters.RecordsPerPage);

				// set page count headers
				SetHeaderCounts(recordCount,
					(queryParameters.RecordsPerPage > 0 ? queryParameters.RecordsPerPage :
						MetadataFilters.DefaultRecordsPerPage));

				// map to returnable data type and return
				return Ok(results);
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
		/// Fetch a specific detection based on a unique ID.
		/// </summary>
		/// <param name="id">Detection's unique ID</param>
		/// <response code="200">Returns the detection.</response>
		/// <response code="400">If the request was malformed (missing parameters).</response>
		/// <response code="404">If the detection defined by the unique Id could not be found.</response>
		/// <response code="500">If there is an internal error reading or processing the data from CosmosDB.</response>
		[HttpGet("{id}")]
		[ProducesResponseType(typeof(Detection), 200)]
		[ProducesResponseType(400)]
		[ProducesResponseType(404)]
		[ProducesResponseType(500)]
		public async Task<IActionResult> GetByIdAsync(string id)
		{
			try
			{
				if (string.IsNullOrWhiteSpace(id))
					throw new ArgumentNullException("id");

				var metadata = await _repository.GetByIdAsync(id);

				if (metadata == null)
					return NotFound();

				return Ok(DetectionProcessors.ToDetection(metadata));
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
		/// List all AI/ML generated detections that have not yet been reviewed (confirmed or rejected) by a human moderator.
		/// </summary>
		/// <response code="200">Returns the list of unreviewed detections.</response>
		/// <response code="204">If there are no detections for the specified timeframe.</response>
		/// <response code="400">If the request was malformed (missing parameters).</response>
		/// <response code="500">If there is an internal error reading or processing the data from CosmosDB.</response>
		[HttpGet]
		[Route("unreviewed")]
		[ProducesResponseType(typeof(IEnumerable<Detection>), 200)]
		[ProducesResponseType(204)]
		[ProducesResponseType(400)]
		[ProducesResponseType(500)]
		public IActionResult GetUnreviewed([FromQuery] DetectionQueryParameters queryParameters)
		{
			try
			{
				if (string.IsNullOrWhiteSpace(queryParameters.Timeframe))
					throw new ArgumentNullException("Timeframe");

				if (queryParameters.DateFrom > queryParameters.DateTo)
					throw new Exception("From Date should be less than To date");

				if (string.IsNullOrWhiteSpace(queryParameters.SortBy))
					throw new ArgumentNullException("SortBy");

				if (string.IsNullOrWhiteSpace(queryParameters.SortOrder))
					throw new ArgumentNullException("SortOrder");

				if (string.IsNullOrWhiteSpace(queryParameters.Location))
					throw new ArgumentNullException("Location");
				
				if (queryParameters.Page == 0)
					throw new ArgumentNullException("Page");

				if (queryParameters.RecordsPerPage == 0)
					throw new ArgumentNullException("RecordsPerPage");

				// start with all records
				var queryable = _repository.GetAll();

				// apply reviewed status
				MetadataFilters.ApplyReviewedFilter(ref queryable, false);

                // apply location filter
                if (queryParameters.Location.ToLower() != "all")
                {
                    MetadataFilters.ApplyLocationFilter(ref queryable, queryParameters.Location);
                }

				// apply timeframe filter
				MetadataFilters.ApplyTimeframeFilter(ref queryable, queryParameters.Timeframe, queryParameters.DateFrom, queryParameters.DateTo);

				// If no detections found
				if (queryable == null || queryable.Count() == 0)
				{
					return NoContent();
				}

				// total number of records
				double recordCount = queryable.Count();

				var results = queryable
					.Select(x => DetectionProcessors.ToDetection(x)).ToList();

				// NOTE: Have to apply SortBy timestamp and pagination filter after
				//       executing the select because of how EF for Cosmos deals with DateTime.
				//       Had to convert from string (how stored in Cosmos) to DateTime in order to apply the
				//       select, but that messed up the SortBy since Cosmos is expecting a string.

				// apply sort filter
				if (queryParameters.SortBy.ToLower() == "confidence")
					DetectionFilters.ApplyConfidenceSortFilter(ref results, queryParameters.SortOrder);
				else if (queryParameters.SortBy.ToLower() == "timestamp")
					DetectionFilters.ApplyTimestampSortFilter(ref results, queryParameters.SortOrder);

				// apply pagination filter
				DetectionFilters.ApplyPaginationFilter(ref results, queryParameters.Page, queryParameters.RecordsPerPage);


                // set page count headers
                SetHeaderCounts(recordCount,
					(queryParameters.RecordsPerPage > 0 ? queryParameters.RecordsPerPage :
						MetadataFilters.DefaultRecordsPerPage));

				// map to returnable data type and return
				return Ok(results);
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
		/// List all AI/ML generated detections that have been reviewed by a human moderator and have confirmed whale sounds.
		/// </summary>
		/// <response code="200">Returns the list of confirmed detections.</response>
		/// <response code="204">If there are no detections for the specified timeframe.</response>
		/// <response code="400">If the request was malformed (missing parameters).</response>
		/// <response code="500">If there is an internal error reading or processing the data from CosmosDB.</response>
		[HttpGet]
		[Route("confirmed")]
		[ProducesResponseType(typeof(IEnumerable<Detection>), 200)]
		[ProducesResponseType(204)]
		[ProducesResponseType(400)]
		[ProducesResponseType(500)]
		public IActionResult GetConfirmed([FromQuery] DetectionQueryParameters queryParameters)
		{
			try
			{
				if (string.IsNullOrWhiteSpace(queryParameters.Timeframe))
					throw new ArgumentNullException("Timeframe");

				if (queryParameters.DateFrom > queryParameters.DateTo)
					throw new Exception("From Date should be less than To date");

				if (string.IsNullOrWhiteSpace(queryParameters.SortBy))
					throw new ArgumentNullException("SortBy");

				if (string.IsNullOrWhiteSpace(queryParameters.SortOrder))
					throw new ArgumentNullException("SortOrder");

				if (string.IsNullOrWhiteSpace(queryParameters.Location))
					throw new ArgumentNullException("Location");

				if (queryParameters.Page == 0)
					throw new ArgumentNullException("Page");

				if (queryParameters.RecordsPerPage == 0)
					throw new ArgumentNullException("RecordsPerPage");

				// start with all records
				var queryable = _repository.GetAll();

				// apply desired status
				MetadataFilters.ApplyReviewedFilter(ref queryable, true);

				// apply desired found state
				MetadataFilters.ApplyFoundFilter(ref queryable, "yes");

				// apply timeframe filter
				MetadataFilters.ApplyTimeframeFilter(ref queryable, queryParameters.Timeframe, queryParameters.DateFrom, queryParameters.DateTo);

				// apply location filter
				if (queryParameters.Location.ToLower() != "all")
				{
					MetadataFilters.ApplyLocationFilter(ref queryable, queryParameters.Location);
				}

				// If no detections found
				if (queryable == null || queryable.Count() == 0)
				{
					return NoContent();
				}

				// total number of records
				double recordCount = queryable.Count();

				var results = queryable
					.Select(x => DetectionProcessors.ToDetection(x)).ToList();

				// apply sort filter
				if (queryParameters.SortBy.ToLower() == "confidence")
					DetectionFilters.ApplyConfidenceSortFilter(ref results, queryParameters.SortOrder);
				else if (queryParameters.SortBy.ToLower() == "timestamp")
					DetectionFilters.ApplyTimestampSortFilter(ref results, queryParameters.SortOrder);

				// apply pagination filter
				DetectionFilters.ApplyPaginationFilter(ref results, queryParameters.Page, queryParameters.RecordsPerPage);

				// set page count headers
				SetHeaderCounts(recordCount,
					(queryParameters.RecordsPerPage > 0 ? queryParameters.RecordsPerPage :
						MetadataFilters.DefaultRecordsPerPage));

				// map to returnable data type and return
				return Ok(results);
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
		/// List all AI/ML generated detections that have been reviewed by a human moderator, but do not have whale sounds.
		/// </summary>
		/// <response code="200">Returns the list of false positive detections.</response>
		/// <response code="204">If there are no detections for the specified timeframe.</response>
		/// <response code="400">If the request was malformed (missing parameters).</response>
		/// <response code="500">If there is an internal error reading or processing the data from CosmosDB.</response>
		[HttpGet]
		[Route("falsepositives")]
		[ProducesResponseType(typeof(IEnumerable<Detection>), 200)]
		[ProducesResponseType(204)]
		[ProducesResponseType(400)]
		[ProducesResponseType(500)]
		public IActionResult GetFalsePositives([FromQuery] DetectionQueryParameters queryParameters)
		{
			try
			{
				if (string.IsNullOrWhiteSpace(queryParameters.Timeframe))
					throw new ArgumentNullException("Timeframe");

				if (queryParameters.DateFrom > queryParameters.DateTo)
					throw new Exception("From Date should be less than To date");

				if (string.IsNullOrWhiteSpace(queryParameters.SortBy))
					throw new ArgumentNullException("SortBy");

				if (string.IsNullOrWhiteSpace(queryParameters.SortOrder))
					throw new ArgumentNullException("SortOrder");

				if (string.IsNullOrWhiteSpace(queryParameters.Location))
					throw new ArgumentNullException("Location");

				if (queryParameters.Page == 0)
					throw new ArgumentNullException("Page");

				if (queryParameters.RecordsPerPage == 0)
					throw new ArgumentNullException("RecordsPerPage");

				// start with all records
				var queryable = _repository.GetAll();

				// apply desired status
				MetadataFilters.ApplyReviewedFilter(ref queryable, true);

				// apply desired found state
				MetadataFilters.ApplyFoundFilter(ref queryable, "no");

				// apply timeframe filter
				MetadataFilters.ApplyTimeframeFilter(ref queryable, queryParameters.Timeframe, queryParameters.DateFrom, queryParameters.DateTo);

				// apply location filter
				if (queryParameters.Location.ToLower() != "all")
				{
					MetadataFilters.ApplyLocationFilter(ref queryable, queryParameters.Location);
				}

				// If no detections found
				if (queryable == null || queryable.Count() == 0)
				{
					return NoContent();
				}

				// total number of records
				double recordCount = queryable.Count();

				var results = queryable
					.Select(x => DetectionProcessors.ToDetection(x)).ToList();

				// apply sort filter
				if (queryParameters.SortBy.ToLower() == "confidence")
					DetectionFilters.ApplyConfidenceSortFilter(ref results, queryParameters.SortOrder);
				else if (queryParameters.SortBy.ToLower() == "timestamp")
					DetectionFilters.ApplyTimestampSortFilter(ref results, queryParameters.SortOrder);

				// apply pagination filter
				DetectionFilters.ApplyPaginationFilter(ref results, queryParameters.Page, queryParameters.RecordsPerPage);

				// set page count headers
				SetHeaderCounts(recordCount,
					(queryParameters.RecordsPerPage > 0 ? queryParameters.RecordsPerPage :
						MetadataFilters.DefaultRecordsPerPage));

				// map to returnable data type and return
				return Ok(results);
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
		/// List all AI/ML generated detections that have been reviewed by a human moderator, but whale sounds could not be conclusively confirmed or denied.
		/// </summary>
		/// <response code="200">Returns the list of unknown detections.</response>
		/// <response code="204">If there are no detections for the specified timeframe.</response>
		/// <response code="400">If the request was malformed (missing parameters).</response>
		/// <response code="500">If there is an internal error reading or processing the data from CosmosDB.</response>
		[HttpGet]
		[Route("unknowns")]
		[ProducesResponseType(typeof(IEnumerable<Detection>), 200)]
		[ProducesResponseType(204)]
		[ProducesResponseType(400)]
		[ProducesResponseType(500)]
		public IActionResult GetUnknowns([FromQuery] DetectionQueryParameters queryParameters)
		{
			try
			{
				if (string.IsNullOrWhiteSpace(queryParameters.Timeframe))
					throw new ArgumentNullException("Timeframe");

				if (queryParameters.DateFrom > queryParameters.DateTo)
					throw new Exception("From Date should be less than To date");

				if (string.IsNullOrWhiteSpace(queryParameters.SortBy))
					throw new ArgumentNullException("SortBy");

				if (string.IsNullOrWhiteSpace(queryParameters.SortOrder))
					throw new ArgumentNullException("SortOrder");

				if (string.IsNullOrWhiteSpace(queryParameters.Location))
					throw new ArgumentNullException("Location");

				if (queryParameters.Page == 0)
					throw new ArgumentNullException("Page");

				if (queryParameters.RecordsPerPage == 0)
					throw new ArgumentNullException("RecordsPerPage");

				// start with all records
				var queryable = _repository.GetAll();

				// apply desired status
				MetadataFilters.ApplyReviewedFilter(ref queryable, true);

				// apply desired found state
				MetadataFilters.ApplyFoundFilter(ref queryable, "don't know");

				// apply timeframe filter
				MetadataFilters.ApplyTimeframeFilter(ref queryable, queryParameters.Timeframe, queryParameters.DateFrom, queryParameters.DateTo);

				// apply location filter
				if (queryParameters.Location.ToLower() != "all")
				{
					MetadataFilters.ApplyLocationFilter(ref queryable, queryParameters.Location);
				}

				// If no detections found
				if (queryable == null || queryable.Count() == 0)
				{
					return NoContent();
				}

				// total number of records
				double recordCount = queryable.Count();

				var results = queryable
					.Select(x => DetectionProcessors.ToDetection(x)).ToList();

				// apply sort filter
				if (queryParameters.SortBy.ToLower() == "confidence")
					DetectionFilters.ApplyConfidenceSortFilter(ref results, queryParameters.SortOrder);
				else if (queryParameters.SortBy.ToLower() == "timestamp")
					DetectionFilters.ApplyTimestampSortFilter(ref results, queryParameters.SortOrder);

				// apply pagination filter
				DetectionFilters.ApplyPaginationFilter(ref results, queryParameters.Page, queryParameters.RecordsPerPage);

				// set page count headers
				SetHeaderCounts(recordCount,
					(queryParameters.RecordsPerPage > 0 ? queryParameters.RecordsPerPage :
						MetadataFilters.DefaultRecordsPerPage));

				// map to returnable data type and return
				return Ok(results);
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
		/// Updates the detection with information provided by a human moderator.
		/// </summary>
		/// <param name="id">Detection's unique Id</param>
		/// <param name="detectionUpdate">Detection's update payload</param>
		/// <response code="200">Returns the contents of the updated detection.</response>
		/// <response code="400">If the request was malformed (missing parameters or payload).</response>
		/// <response code="404">Indicates the detection was not found to update.</response>
		/// <response code="500">If there is an internal error updating or processing the data from CosmosDB.</response>
		[HttpPut("{id}")]
		[ProducesResponseType(typeof(Detection), 200)]
		[ProducesResponseType(400)]
		[ProducesResponseType(404)]
		[ProducesResponseType(500)]
		public async Task<IActionResult> Put(string id, [FromBody] DetectionUpdate detectionUpdate)
		{
			try
			{
				if (string.IsNullOrWhiteSpace(id))
					throw new ArgumentNullException("id");

				if (detectionUpdate == null)
					throw new ArgumentNullException("postedDetection");

				var metadata = await _repository.GetByIdAsync(id);

				if (metadata == null)
					return NotFound();

				metadata.comments = detectionUpdate.Comments;
				metadata.moderator = detectionUpdate.Moderator;
				metadata.dateModerated = detectionUpdate.Moderated.ToUniversalTime().ToString("yyyy-MM-ddTHH:mm:ssZ");
				metadata.reviewed = detectionUpdate.Reviewed;
				metadata.SRKWFound = (string.IsNullOrWhiteSpace(detectionUpdate.Found)) ? "no" : detectionUpdate.Found.ToLower();

				// Normalize the tags
				if (!string.IsNullOrWhiteSpace(detectionUpdate.Tags))
				{
					var working = detectionUpdate.Tags.Replace(",", ";");

					var tagList = new List<string>();
					tagList.AddRange(working.Split(';').ToList().Select(x => x.Trim()));
					metadata.tags = string.Join(";", tagList);
				}
				else
				{
					metadata.tags = string.Empty;
				}

				await _repository.CommitAsync();

				return Ok(DetectionProcessors.ToDetection(metadata));
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
