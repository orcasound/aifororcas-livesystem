using System;

namespace Orcasound.Shared.Entities
{
	public class UpdateRequest
	{
		public string Id { get; set; }
		public string Comments { get; set; }
		public string Tags { get; set; }
		public string Moderator { get; set; }
		public DateTime DateModerated { get; set; }
		public string Status { get; set; }
		public string Found { get; set; }
	}
}
