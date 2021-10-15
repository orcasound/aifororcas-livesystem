using AIForOrcas.DTO.API.Tags;
using System.Collections.Generic;
using System.Threading.Tasks;

namespace AIForOrcas.Client.BL.Services
{
    public interface ITagService
    {
        Task<List<string>> GetUniqueTagsAsync();
        Task<int> UpdateTagAsync(TagUpdate payload);
        Task<int> DeleteTagAsync(string tag);
    }
}
