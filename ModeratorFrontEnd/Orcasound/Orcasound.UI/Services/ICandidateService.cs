using Orcasound.Shared.Entities;
using System.Collections.Generic;

namespace Orcasound.UI.Services
{
	public interface ICandidateService
	{
		IEnumerable<Candidate> GetAll();
	}
}