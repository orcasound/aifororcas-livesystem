using Newtonsoft.Json.Linq;
using NotificationSystem.Models;
using NotificationSystem.Template;
using System;
using System.Collections.Generic;
using Microsoft.Extensions.Logging;
using Moq;

namespace NotificationSystem.Tests.Unit
{
    public class EmailTemplateTests
    {
        /// <summary>
        /// Tests that GetSubscriberEmailBody generates correct map URIs for various locations
        /// by verifying the generated HTML contains the expected image URLs.
        /// Uses mocked OrcasiteHelper to simulate production behavior.
        /// </summary>
        [Theory]
        [InlineData("Sunset Bay", "sunset-bay.jpg")]
        [InlineData("Mast Center", "mast-center.jpg")]
        [InlineData("North San Juan Channel", "north-sjc.jpg")]
        [InlineData("Point Robinson", "point-robinson.jpg")]
        [InlineData("Bush Point", "bush-point.jpg")]
        [InlineData("Haro Strait", "haro-strait.jpg")]
        [InlineData("Port Townsend", "port-townsend.jpg")]
        [InlineData("Orcasound Lab", "orcasound-lab.jpg")]
        public void GetSubscriberEmailBody_GeneratesCorrectMapUri_ForLocationName(string locationName, string expectedFileName)
        {
            // Arrange
            var messages = new List<JObject>
            {
                JObject.FromObject(new
                {
                    timestamp = DateTime.UtcNow,
                    location = new
                    {
                        name = locationName,
                        latitude = 48.123,
                        longitude = -122.456,
                        id = "test_location"
                    },
                    moderator = "Test Moderator",
                    comments = "Test comments"
                })
            };

            // Mock OrcasiteHelper to simulate production behavior
            var mockOrcasiteHelper = new Mock<OrcasiteHelper>(
                new Mock<ILogger<OrcasiteHelper>>().Object,
                new System.Net.Http.HttpClient()
            );
            
            // Setup slug mappings based on actual Orcasite feeds
            mockOrcasiteHelper.Setup(x => x.GetSlugByLocationName(It.IsAny<string>()))
                .Returns<string>(name =>
                {
                    // Return actual slugs from Orcasite for locations where they differ from simple transformation
                    if (name == "North San Juan Channel") return "north-sjc";
                    // For other locations, return null to fall back to simple transformation
                    return null;
                });

            string expectedMapUrl = $"https://orcanotificationstorage.blob.core.windows.net/images/{expectedFileName}";

            // Act - with OrcasiteHelper as in production
            string emailBody = EmailTemplate.GetSubscriberEmailBody(messages, mockOrcasiteHelper.Object);

            // Assert
            Assert.Contains(expectedMapUrl, emailBody);
        }

        /// <summary>
        /// Tests that location names with multiple words are correctly converted with hyphens in URIs.
        /// </summary>
        [Fact]
        public void GetSubscriberEmailBody_HandlesMultipleSpacesCorrectly()
        {
            // Arrange
            var messages = new List<JObject>
            {
                JObject.FromObject(new
                {
                    timestamp = DateTime.UtcNow,
                    location = new
                    {
                        name = "North San Juan Channel",
                        latitude = 48.591294,
                        longitude = -123.058779,
                        id = "rpi_north_sjc"
                    },
                    moderator = "Test Moderator",
                    comments = "Test comments"
                })
            };

            // Mock OrcasiteHelper to return the correct slug
            var mockOrcasiteHelper = new Mock<OrcasiteHelper>(
                new Mock<ILogger<OrcasiteHelper>>().Object,
                new System.Net.Http.HttpClient()
            );
            mockOrcasiteHelper.Setup(x => x.GetSlugByLocationName("North San Juan Channel"))
                .Returns("north-sjc");

            // Act - with OrcasiteHelper as in production
            string emailBody = EmailTemplate.GetSubscriberEmailBody(messages, mockOrcasiteHelper.Object);

            // Assert - the URI should use "north-sjc" from OrcasiteHelper
            Assert.Contains("north-sjc.jpg", emailBody);
            Assert.DoesNotContain("north-san-juan-channel.jpg", emailBody);
            // The location name should still display with spaces
            Assert.Contains("North San Juan Channel", emailBody);
        }

        /// <summary>
        /// Tests that the fallback behavior works when OrcasiteHelper is not available.
        /// </summary>
        [Fact]
        public void GetSubscriberEmailBody_FallsBackToSimpleTransformation_WhenOrcasiteHelperNotProvided()
        {
            // Arrange
            var messages = new List<JObject>
            {
                JObject.FromObject(new
                {
                    timestamp = DateTime.UtcNow,
                    location = new
                    {
                        name = "Sunset Bay",
                        latitude = 47.86497296593844,
                        longitude = -122.33393605795372,
                        id = "rpi_sunset_bay"
                    },
                    moderator = "Test Moderator",
                    comments = "Test comments"
                })
            };

            // Act - without OrcasiteHelper, it falls back to simple transformation
            string emailBody = EmailTemplate.GetSubscriberEmailBody(messages, null);

            // Assert - should use simple transformation
            Assert.Contains("sunset-bay.jpg", emailBody);
            Assert.Contains("Sunset Bay", emailBody);
        }

        /// <summary>
        /// Tests that the email body contains all required sections and location information.
        /// </summary>
        [Fact]
        public void GetSubscriberEmailBody_ContainsAllRequiredSections()
        {
            // Arrange
            var testTimestamp = new DateTime(2025, 1, 15, 10, 30, 0, DateTimeKind.Utc);
            var messages = new List<JObject>
            {
                JObject.FromObject(new
                {
                    timestamp = testTimestamp,
                    location = new
                    {
                        name = "Sunset Bay",
                        latitude = 47.86497296593844,
                        longitude = -122.33393605795372,
                        id = "rpi_sunset_bay"
                    },
                    moderator = "Jane Doe",
                    comments = "Clear SRKW calls detected"
                })
            };

            // Act - without OrcasiteHelper, it falls back to simple transformation
            string emailBody = EmailTemplate.GetSubscriberEmailBody(messages, null);

            // Assert
            Assert.Contains("Southern Resident Killer Whale Detected", emailBody);
            Assert.Contains("Sunset Bay", emailBody);
            Assert.Contains("47.86497296593844", emailBody);
            Assert.Contains("-122.33393605795372", emailBody);
            Assert.Contains("Jane Doe", emailBody);
            Assert.Contains("Clear SRKW calls detected", emailBody);
            Assert.Contains("https://orcanotificationstorage.blob.core.windows.net/images/sunset-bay.jpg", emailBody);
        }

        /// <summary>
        /// Tests that GetSubscriberEmailBody uses OrcasiteHelper to lookup the correct slug when provided.
        /// </summary>
        [Fact]
        public void GetSubscriberEmailBody_UsesOrcasiteHelperSlug_WhenProvided()
        {
            // Arrange
            var messages = new List<JObject>
            {
                JObject.FromObject(new
                {
                    timestamp = DateTime.UtcNow,
                    location = new
                    {
                        name = "North San Juan Channel",
                        latitude = 48.591294,
                        longitude = -123.058779,
                        id = "rpi_north_sjc"
                    },
                    moderator = "Test Moderator",
                    comments = "Test comments"
                })
            };

            // Mock OrcasiteHelper that returns the actual slug
            var mockOrcasiteHelper = new Mock<OrcasiteHelper>(
                new Mock<ILogger<OrcasiteHelper>>().Object,
                new System.Net.Http.HttpClient()
            );
            mockOrcasiteHelper.Setup(x => x.GetSlugByLocationName(It.IsAny<string>()))
                .Returns<string>(locationName => 
                {
                    if (locationName == "North San Juan Channel") return "north-sjc";
                    return null;
                });
            
            // Act
            string emailBody = EmailTemplate.GetSubscriberEmailBody(messages, mockOrcasiteHelper.Object);

            // Assert - should use "north-sjc" from OrcasiteHelper, not "north-san-juan-channel"
            Assert.Contains("north-sjc.jpg", emailBody);
            Assert.DoesNotContain("north-san-juan-channel.jpg", emailBody);
            Assert.Contains("North San Juan Channel", emailBody);
        }
    }
}
