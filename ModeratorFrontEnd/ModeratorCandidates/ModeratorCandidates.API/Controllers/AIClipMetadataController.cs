using Microsoft.AspNetCore.Http;
using Microsoft.AspNetCore.Mvc;
using ModeratorCandidates.API.Services;
using ModeratorCandidates.Shared.Models;
using System;
using System.Collections.Generic;
using System.Linq;

namespace ModeratorCandidates.API.Controllers
{
    // TODO: Many design decisions still need to be made and implemented:
    // 1. What is the authentication/authorization scheme here going forward?
    // 2. Do the calls to the service need to be async? That will probably depend on how the data component is implemented.
    // 3. Do we need to provide a filtering component to allow moderators to further refine their experience (i.e. restrict to certain days, certain hydrophone locations, etc.)?
    // 4. How many moderators will there be? Do we need to be concerned about overlapping/conflicting sessions?

    [Route("api/aiclipmetadata")]
    [ApiController]
    public class AIClipMetadataController : ControllerBase
    {
        private readonly AIClipMetadataService service;

        public AIClipMetadataController(AIClipMetadataService service)
        {
            this.service = service;
        }

        [HttpGet]
        [ProducesResponseType(typeof(IEnumerable<AIClipMetadata>), 200)]
        [ProducesResponseType(500)]
        public IActionResult Get([FromQuery] Pagination pagination)
        {
            try
            {
                var queryable = service.GetAll();

                double count = queryable.Count();

                queryable = queryable.OrderBy(x => x.timestamp).ThenBy(x => x.id);

                queryable = queryable
                    .Skip((pagination.Page - 1) * pagination.RecordsPerPage)
                    .Take(pagination.RecordsPerPage);

                double totalAmountPages = Math.Ceiling(count / pagination.RecordsPerPage);

                HttpContext.Response.Headers.Add("totalAmountPages", totalAmountPages.ToString());

                return Ok(queryable.ToList());
            }
            catch (Exception ex)
            {
                return StatusCode(StatusCodes.Status500InternalServerError, ex.Message);
            }
        }

        [HttpGet]
        [Route("unreviewed")]
        [ProducesResponseType(typeof(IEnumerable<AIClipMetadata>), 200)]
        [ProducesResponseType(500)]
        public IActionResult GetUnreviewed([FromQuery] Pagination pagination)
        {
            try
            {
                var queryable = service.GetAll();

                queryable = queryable.Where(x => x.status == "Unreviewed");

                double count = queryable.Count();

                queryable = queryable.OrderBy(x => x.timestamp).ThenBy(x => x.id);

                queryable = queryable
                    .Skip((pagination.Page - 1) * pagination.RecordsPerPage)
                    .Take(pagination.RecordsPerPage);

                double totalAmountPages = Math.Ceiling(count / pagination.RecordsPerPage);

                HttpContext.Response.Headers.Add("totalAmountPages", totalAmountPages.ToString());

                return Ok(queryable.ToList());
            }
            catch (Exception ex)
            {
                return StatusCode(StatusCodes.Status500InternalServerError, ex.Message);
            }
        }

        [HttpGet("{id}")]
        [ProducesResponseType(typeof(AIClipMetadata), 200)]
        [ProducesResponseType(404)]
        [ProducesResponseType(500)]
        public IActionResult GetById(string id)
        {
            try
            {
                if (string.IsNullOrWhiteSpace(id))
                {
                    return BadRequest("No id provided");
                }
                var record = service.GetById(id);

                if (record == null)
                {
                    return NotFound();
                }

                return Ok(record);
            }
            catch (Exception ex)
            {
                return StatusCode(StatusCodes.Status500InternalServerError, ex.Message);
            }
        }

        [HttpPut("{id}")]
        public IActionResult Put(string id, [FromForm] AIClipMetadataReviewResult result)
        {
            var record = service.GetById(id);

            if (record == null) { return NotFound(); }

            record.comments = result.comments;
            record.status = result.status;
            record.moderator = result.moderator;
            record.dateModerated = result.dateModerated;
            record.tags = result.tags;

            // TODO: Would assume additional workflow goes here if SRKW are found
            //       like kicking off email/push notification, etc.
            //       Would also assume additional workflow if not found like
            //       training the model

            return NoContent();
        }
    }
}