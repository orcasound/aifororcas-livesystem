using Microsoft.AspNetCore.Http;

namespace ModeratorCandidates.API.Helpers
{
	public class UrlHelper
	{
		private readonly IHttpContextAccessor contextAccessor;

		public UrlHelper(IHttpContextAccessor contextAccessor)
		{
			this.contextAccessor = contextAccessor;
		}

		public string FullURL()
		{
			var request = contextAccessor.HttpContext.Request;

			var absoluteUri = string.Concat(
						request.Scheme,
						"://",
						request.Host.ToUriComponent(),
						request.PathBase.ToUriComponent(),
						request.Path.ToUriComponent(),
						request.QueryString.ToUriComponent());
			return absoluteUri;
		}

		public string BaseURL()
		{
			var request = contextAccessor.HttpContext.Request;

			var baseUri = string.Concat(
						request.Scheme,
						"://",
						request.Host.ToUriComponent(),
						request.PathBase.ToUriComponent());
			return baseUri;
		}
	}
}
