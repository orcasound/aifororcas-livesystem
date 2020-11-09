using Microsoft.AspNetCore.Components.Authorization;
using System.Threading.Tasks;

namespace AIForOrcas.Client.BL.Helpers
{
	public class IdentityHelper
	{
		private readonly AuthenticationStateProvider _authenticationStateProvider;

		public IdentityHelper(AuthenticationStateProvider authenticationStateProvider)
		{
			_authenticationStateProvider = authenticationStateProvider;
		}

		public async Task<string> GetName()
		{
			if (_authenticationStateProvider != null)
			{
				var authState = await _authenticationStateProvider.GetAuthenticationStateAsync();
				var user = authState.User;

				if (user.Identity.IsAuthenticated)
				{
					var name = user.FindFirst(c => c.Type == "name")?.Value;
					var identity = user.Identity.Name;
					return string.IsNullOrWhiteSpace(name) ? identity : name;
				}
			}

			return string.Empty;
		} 
	}
}
