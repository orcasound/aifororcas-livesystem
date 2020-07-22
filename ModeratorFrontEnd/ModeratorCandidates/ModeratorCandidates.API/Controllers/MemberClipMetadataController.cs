using Microsoft.AspNetCore.Http;
using Microsoft.AspNetCore.Mvc;
using ModeratorCandidates.Shared.Models;

namespace ModeratorCandidates.API.Controllers
{
    // TODO: This is currently a placeholder but the idea is to provide a mechanism for citizen scientist to
    // nominate clips of audio they have heard so a moderator can review. Would probably have the same basic endpoints as
    // AIClipMetadata 

    [Route("api/memberclipmetadata")]
    [ApiController]
    public class MemberClipMetadataController : ControllerBase
    {

        [HttpGet]
        [ProducesResponseType(500)]
        public IActionResult Get([FromQuery] Pagination pagination)
        {
            // TODO: Return some real content from here
            return StatusCode(StatusCodes.Status501NotImplemented, "Under construction");
        }

        [HttpGet]
        [Route("unreviewed")]
        [ProducesResponseType(500)]
        public IActionResult GetUnreviewed([FromQuery] Pagination pagination)
        {
            // TODO: Return some real content from here
            return StatusCode(StatusCodes.Status501NotImplemented, "Under construction");
        }

        [HttpGet("{id}")]
        [ProducesResponseType(501)]
        public IActionResult GetById([FromQuery]string id)
        {
            // TODO: Return some real content from here
            return StatusCode(StatusCodes.Status501NotImplemented, "Under construction");
        }

        [HttpPut("{id}")]
        public IActionResult Put(string id)
        {
            // TODO: Return some real content from here
            return StatusCode(StatusCodes.Status501NotImplemented, "Under construction");
        }
    }
}