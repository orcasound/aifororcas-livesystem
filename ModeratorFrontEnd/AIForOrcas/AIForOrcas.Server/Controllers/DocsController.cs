using Microsoft.AspNetCore.Authorization;
using Microsoft.AspNetCore.Mvc;

namespace AIForOrcas.Server.Controllers
{
	[Route("")]
	[ApiExplorerSettings(IgnoreApi = true)]
	public class DocsController : Controller
	{
        [Route("docs"), HttpGet]
        [AllowAnonymous]
        public IActionResult ReDoc()
        {
            return View();
        }

        [Route(""), HttpGet]
        [AllowAnonymous]
        public IActionResult Swagger()
        {
            return Redirect("~/swagger");
        }
    }
}
