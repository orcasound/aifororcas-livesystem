using Orcasound.Shared.Entities;
using System;
using System.Collections.Generic;

namespace Orcasound.UI.Services
{
	public class CandidateService : ICandidateService
	{
		private Random rnd = new Random();
		private List<string> nodes = new List<string>() { "Port Townsend", "Bush Point", "Haro Strait" };
		private List<string> descriptions = new List<string>() { "I think I heard something!", "Is this a whale?", "Darn. It's just a cargo ship" };

		public IEnumerable<Candidate> GetAll()
		{
			var results = new List<Candidate>();

			for (int i = 0; i < 100; i++)
			{
				var candidate = new Candidate();

				candidate.Id = rnd.Next(100, 200);

				DateTime minDt = new DateTime(2019, 1, 1, 10, 0, 0);
				DateTime maxDt = new DateTime(2020, 12, 31, 17, 0, 0);
				List<DateTime> myDates = new List<DateTime>();
				//Random.Next in .NET is non-inclusive to the upper bound (@NickLarsen)
				int minutesDiff = Convert.ToInt32(maxDt.Subtract(minDt).TotalMinutes + 1);

				int r = rnd.Next(1, minutesDiff);
				candidate.Timestamp = minDt.AddMinutes(r);

				r = rnd.Next(0, 3);
				candidate.Node = nodes[r];

				r = rnd.Next(0, 2);
				if(r == 0)
				{
					candidate.Source = "AI";
					candidate.Probability = rnd.NextDouble() * (99.99 - 75.00) + 75.00;
				}
				else
				{
					candidate.Source = "Citizen";
					r = rnd.Next(0, 3);
					candidate.Description = descriptions[r];
				}

				r = rnd.Next(0, 3);
				if(r == 0)
				{
					candidate.Tags = new List<string>() { "Calls" };
				}
				if(r == 1)
				{
					candidate.Tags = new List<string>() { "Calls", "Whistles" };
				}
				if (r == 2)
				{
					candidate.Tags = new List<string>() { "Clicks", "Whistles" };
				}

				results.Add(candidate);
			}

			return results;
		}
	}
}
