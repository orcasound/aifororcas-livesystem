using System.Collections.Generic;
using System.Threading.Tasks;

namespace AIForOrcas.Client.BL.Services
{
    public interface ITagService
    {
        Task<List<string>> GetUniqueTagsAsync();
    }
}
