using AIForOrcas.DTO.API.Tags;
using AIForOrcas.Server.BL.Services;
using Microsoft.AspNetCore.Http;
using Microsoft.AspNetCore.Mvc;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;

namespace AIForOrcas.Server.Controllers
{
    [Produces("application/json")]
    [Route("api/tags")]
    [ApiController]
    public class TagsController : ControllerBase
    {
        private readonly MetadataRepository _repository;

        public TagsController(MetadataRepository repository)
        {
            _repository = repository;
        }

        /// <summary>
        /// Gets a unique list of Tags.
        /// </summary>
        /// <response code="200">Returns the list of unique Tags.</response>
        /// <response code="204">Indicates there were no Tags to return.</response>
        /// <response code="500">If there is an internal error updating or processing the data from CosmosDB.</response>
        [HttpGet]
        [ProducesResponseType(typeof(IEnumerable<string>), 200)]
        [ProducesResponseType(204)]
        [ProducesResponseType(500)]
        public IActionResult GetAll()
        {
            try
            {
                var rawTags = _repository.GetAllTags().ToList();

                if (rawTags == null || rawTags.Count() == 0)
                    return NoContent();

                var uniqueTags = new List<string>();

                rawTags.ForEach(x => uniqueTags.AddRange(x.Split(";")));

                uniqueTags = uniqueTags.Distinct().OrderBy(x => x).ToList();

                return Ok(uniqueTags);
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
        /// Updates all occurrences of the old Tag with the new Tag.
        /// </summary>
        /// <param name="tagUpdate">Tag update payload</param>
        /// <response code="200">Returns the number of records updated.</response>
        /// <response code="400">If the request was malformed (missing parameters or payload).</response>
        /// <response code="404">Indicates there were no matching records to update.</response>
        /// <response code="500">If there is an internal error updating or processing the data from CosmosDB.</response>
        [HttpPut]
        [ProducesResponseType(200)]
        [ProducesResponseType(400)]
        [ProducesResponseType(404)]
        [ProducesResponseType(500)]
        public async Task<IActionResult> Put([FromBody] TagUpdate tagUpdate)
        {
            try
            {
                if (tagUpdate == null)
                    throw new ArgumentNullException("tagUpdate");

                var detectionsToUpdate = _repository.GetAllWithTag(tagUpdate.OldTag).ToList();

                if (detectionsToUpdate.Count() == 0)
                    return NoContent();

                foreach(var detection in detectionsToUpdate)
                {
                    detection.tags = detection.tags.Replace(tagUpdate.OldTag, tagUpdate.NewTag);
                }

                await _repository.CommitAsync();

                return Ok(detectionsToUpdate.Count());
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
        /// Delete a tag from the database.
        /// </summary>
        /// <param name="tag">Tag to delete</param>
        /// <response code="200">Returns the number of records updated.</response>
        /// <response code="400">If the request was malformed (missing parameters or payload).</response>
        /// <response code="404">Indicates there were no matching records to update.</response>
        /// <response code="500">If there is an internal error updating or processing the data from CosmosDB.</response>
        [HttpDelete]
        [ProducesResponseType(200)]
        [ProducesResponseType(400)]
        [ProducesResponseType(404)]
        [ProducesResponseType(500)]
        public async Task<IActionResult> Delete(string tag)
        {
            try
            {
                if (string.IsNullOrWhiteSpace(tag))
                    throw new ArgumentNullException("tag");

                var detectionsToUpdate = _repository.GetAllWithTag(tag).ToList();

                if (detectionsToUpdate.Count() == 0)
                    return NoContent();

                foreach (var detection in detectionsToUpdate)
                {
                    var split = detection.tags.Split(";").ToList();
                    split.Remove(tag);
                    detection.tags = string.Join(";", split);
                }

                await _repository.CommitAsync();

                return Ok(detectionsToUpdate.Count());
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
