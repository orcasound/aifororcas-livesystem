using Orcasound.Shared.Entities;
using System.Collections.Generic;
using System.Threading.Tasks;

namespace Orcasound.UI.Services
{
	public interface ICandidateService
	{
		Task<PaginatedResponse<List<Candidate>>> GetCandidates(Pagination pagination);
		Task<PaginatedResponse<List<Candidate>>> GetUnreviewedCandidates(Pagination pagination);
		Task UpdateCandidate(UpdateRequest request);
	}
}