﻿using Microsoft.AspNetCore.Mvc.ModelBinding;
using System.Runtime.Serialization;
using System.Text.Json.Serialization;

namespace AIForOrcas.DTO
{
	/// <summary>
	/// Query parameters for metrics endpoint.
	/// </summary>
	[DataContract]
	public class MetricsFilterDTO : IFilterOptions
	{
		/// <summary>
		/// Timeframe for the record set (last 30m, 3h, 6h, 24h, 1w, 30d, all).
		/// </summary>
		/// <example>all</example>
		[DataMember]
		public string Timeframe { get; set; }

		[BindNever]
		public virtual string QueryString => $"timeframe={Timeframe}";
	}
}
