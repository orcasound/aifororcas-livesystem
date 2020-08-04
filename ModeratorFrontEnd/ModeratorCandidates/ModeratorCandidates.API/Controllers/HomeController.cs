using Microsoft.AspNetCore.Mvc;

namespace ModeratorCandidates.API.Controllers
{
	public class HomeController : Controller
	{
		[Route(""), HttpGet]
		[ApiExplorerSettings(IgnoreApi = true)]
		public RedirectResult RedirectToSwaggerUi()
		{
			return Redirect("/swagger/");
		}

	}
}