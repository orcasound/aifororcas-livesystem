using AIForOrcas.Server.BL.Context;
using AIForOrcas.Server.BL.Models.CosmosDB;
using Microsoft.EntityFrameworkCore;
using System;
using System.Collections.Generic;
using System.Data;
using System.Threading.Tasks;

namespace AIForOrcas.Server.BL.Services
{
	public class MetadataRepository
	{
		private readonly ApplicationDbContext _db;

		public MetadataRepository(ApplicationDbContext db)
		{
			_db = db;
		}

		public async Task<IEnumerable<Metadata>> GetAllAsync()
		{
			return await _db.Metadata.ToListAsync();
		}

		public async Task<Metadata> GetByIdAsync(string id)
		{
			return await _db.Metadata.FirstOrDefaultAsync(x => x.id == id);
		}

		public async Task CommitAsync()
		{
			try
			{
				await _db.SaveChangesAsync();
			}
			catch (Exception ex)
			{
				throw new DataException(ex.Message);
			}
		}
	}
}
