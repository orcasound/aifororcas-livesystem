using Microsoft.EntityFrameworkCore;
using ModeratorCandidates.Shared.Models;
using System;
using System.Collections.Generic;
using System.Threading.Tasks;

namespace ModeratorCandidates.API.Services
{
	public class MetadataRepository
	{
		private readonly ApplicationDbContext _db;

		public MetadataRepository(ApplicationDbContext db)
		{
			_db = db;
		}

		public async Task<IEnumerable<Metadata>> GetAll()
		{
			return await _db.Metadata.ToListAsync();
		}

		public async Task<Metadata> GetById(string id)
		{
			return await _db.Metadata.FirstOrDefaultAsync(x => x.id == id);
		}

		public async Task Commit()
		{
			try
			{
				await _db.SaveChangesAsync();
			}
			catch (Exception ex)
			{
				// TODO: Figure out how we are handling exceptions here.
			}
		}
	}
}
