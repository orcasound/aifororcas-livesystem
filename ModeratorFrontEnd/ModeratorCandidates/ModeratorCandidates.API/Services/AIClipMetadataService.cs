using Microsoft.AspNetCore.Hosting;
using Microsoft.AspNetCore.Http;
using ModeratorCandidates.API.Helpers;
using ModeratorCandidates.Shared.Models;
using Newtonsoft.Json;
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Configuration;

namespace ModeratorCandidates.API.Services
{
	public class AIClipMetadataService
	{
		private readonly IHttpContextAccessor contextAccessor;
		private readonly IWebHostEnvironment webHostEnvironment;

		private readonly List<AIClipMetadata> candidates = new List<AIClipMetadata>();

		public AIClipMetadataService(IHttpContextAccessor contextAccessor, IWebHostEnvironment webHostEnvironment)
		{
			this.contextAccessor = contextAccessor;
			this.webHostEnvironment = webHostEnvironment;
			LoadJsonFilesIntoMemory();
		}

		public IQueryable<AIClipMetadata> GetAll()
		{
			return candidates.AsQueryable();
		}

		public AIClipMetadata GetById(string id)
		{
			return candidates.FirstOrDefault(x => x.id == id);
		}

		#region Helpers
		private void LoadJsonFilesIntoMemory()
		{
			var pathHelper = new PathHelper(webHostEnvironment);
			var dataPath = $"{pathHelper.BasePath()}/Data";

			// verify folder exists
			if (!Directory.Exists(dataPath))
			{
				throw new FileNotFoundException("/Data folder does not exist in project path");
			}

			// iterate the path and get the name of all .json files
			var fileNameList = Directory.GetFiles(dataPath, "*.json", SearchOption.AllDirectories);

			// verify there is data to load
			if (!fileNameList.Any())
			{
				throw new FileNotFoundException("No .json files in /Data folder to import");
			}

			var urlHelper = new UrlHelper(contextAccessor);
			var baseUrl = urlHelper.BaseURL();

			var locationList = new List<AILocation>()
			{
				//new AILocation() { name = "Port Townsend", latitude = 48.088922, longitude = -122.762901 },
				//new AILocation() { name = "Bush Point", latitude = 48.029424, longitude = -122.615434 },
				//new AILocation() { name = "Haro Point", latitude = 48.579333, longitude = -123.178732}
			};

			var rnd = new Random();

			// iterate the candidates and build the data set
			foreach (var fileName in fileNameList)
			{
				// read the file as a json string
				var jsonString = File.ReadAllText(fileName);

				// deserialize the string to an import object
				var import = JsonConvert.DeserializeObject<JsonClipMetadata>(jsonString);

				// calculate needed pieces
				var id = Path.GetFileNameWithoutExtension(fileName);
				var timestamp = ConvertFileNameToTimeStamp(id);

				// calculate lat/long
				// randomly assigning them to locations for testing purposes
				// TODO: update when we actually get sample data from other locations

				var location = locationList[rnd.Next(3)];

				// use that object to build the new candidate object
				// TODO: add reference to real image of candidate (this one is a fake for dev purposes)
				var candidate = new AIClipMetadata
				{
					audioUri = import.uri,
					imageUri = $"{baseUrl}/Images/Sample.PNG",
					timestamp = timestamp,
					id = Path.GetFileNameWithoutExtension(fileName),
					location = location,
					status = "Unreviewed"
				};

				int count = 0;

				//var annotations = import.annotations.Select(annotation => new AIAnnotation()
				//{
				//	id = count++,
				//	startTime = annotation.start_time_s,
				//	duration = annotation.duration_s,
				//	confidence = annotation.confidence,
				//})
				//.ToList();

				//candidate.annotations = annotations;

				// add the candidate to the results list
				candidates.Add(candidate);
			}
		}

		private DateTime ConvertFileNameToTimeStamp(string fileName)
		{
			var components = fileName.Split('_');

			if (components.Length != 9)
			{
				throw new ArgumentException("Passed filename is malformed");
			}

			var timestamp = new DateTime(
				int.Parse(components[3]),
				int.Parse(components[1]),
				int.Parse(components[2]),
				int.Parse(components[4]),
				int.Parse(components[5]),
				int.Parse(components[6]));

			return timestamp;
		}

		#endregion



	}
}
