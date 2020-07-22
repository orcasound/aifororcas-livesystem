using Microsoft.AspNetCore.Hosting;

namespace ModeratorCandidates.API.Helpers
{
	public class PathHelper
	{
		private readonly IWebHostEnvironment webHostEnvironment;

		public PathHelper(IWebHostEnvironment webHostEnvironment)
		{
			this.webHostEnvironment = webHostEnvironment;
		}

		public string BasePath()
		{
			return webHostEnvironment.ContentRootPath;
		}
	}
}
