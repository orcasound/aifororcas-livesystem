# Moderator Front End System

This directory contains two separate moderator front end system implementations:

* AIForOrcas: this implementation is what is currently deployed at https://aifororcas.azurewebsites.net/ and https://aifororcasdetections.azurewebsites.net/swagger/index.html
* OrcaHello: this implementation is intended as the future replacement and is now deployed at https://aifororcasdetections2.azurewebsites.net/swagger/index.html

## AI For Orcas - Moderator Front End System

The moderator front end is responsible for presenting the candidate whale calls to the designated moderators so they can confirm the calls as legitimate or as false positives.

The basic workflow is as follows:
- The notification system sends designated moderators an email indicating there are one or more candidates to review.
- The moderator logs into the moderator front end and is presented with the list of candidate calls including a spectrogram image of the call.
- The moderator then listens to these calls and indicates whether it is a confirmed call, a false positive, or an undetermined call; they can also add sortable tags and provide comments.
- If the call is confirmed, notifications are sent out to the designated subscribers informing them of the presence of whales in the designated area.

## Architecture
The moderator front end solution has two primary components that are both located in the /ModeratorFrontEnd/AIForOrcas folder:

### A Web API to Interact with the CosmosDB backend
This is a standard .NET Core Web API found in the AIForOrcas.Server project and is published to Azure App Service as https://aifororcasdetections.azurewebsites.net/swagger

#### Publish Settings
- Site Url: https://aifororcasdetections.azurewebsites.net/swagger
- Resource group: AIForOrcas
- Configuration: Release
- Target framework: net6.0
- Deployment mode: Self-contained
- Target runtime: win-x86

### A Web Fontend
This is a .NET Core Blazor-based web site found in the AIForOrcas.Client.Web project and is published to Azure App Service as https://aifororcas.azurewebsites.net

#### Publish Settings
- Site Url: https://aifororcas.azurewebsites.net
- Resource group: AIForOrcas
- Configuration: Release
- Target framework: net6.0
- Deployment mode: Self-contained
- Target runtime: win-x86

### Deployment
The AIForOrcas.Server and AIForOrcas.Client.Web projects are built and deployed using GitHub Actions workflows:
* [AIForOrcas.Server.yaml](/.github/workflows/AIForOrcas.Server.yaml)
* [AIForOrcas.Client.Web.yaml](/.github/workflows/AIForOrcas.Client.Web.yaml)

### Additional Projects
The remaining projects in the solution represent code that is shared between the two projects (i.e., DTOs/Models, Business Logic, etc.)

## Release Notes

### 2022-09-21
#### AIForOrcas.Server
**Overall**
- Migrated target framework to .NET 6
- Updated NuGet packages to .NET 6 versions
- Dropped the namespace {} for all classes (a new .NET 6 feature)
- Migrated Program.cs and Startup.cs to hosting in Program.cs (a new .NET 6 feature)
- Moved service injections and configurations into their own extension classes to shorten and cleanup Program.cs
- Added Azure Active Directory authentication

**Swagger**
- Set up authentication for the Swagger documentation so protected endpoints could be accessed
- Cleaned up the Swagger documentation for all the controllers
- Finished descriptions for the classes used in the Swagger documentation
- Removed the Schema list from the bottom the Swagger page

**DetectionsController**
- Set up policy-based authentication for PUT endpoint

**TagsController**
- Set up policy-based authentication for PUT and DELETE endpoints

#### AIForOrcas.Client.Web
**Overall**
- Migrated to .NET 6
- NuGet packages updated to .NET 6 versions
- Migrated Program.cs and Startup.cs to hosting in Program.cs (a new .NET 6 feature)
- Created a global usings file and moved all the using reference into it (a new .NET 6 feature)
- Dropped the namespace {} for all classes (a new .NET 6 feature)
- Set up authentication against the API
- Removed uneeded Azure B2C components
- Moved Configuration settings into their own class and injected into various classes for use
- Moved service injections and configurations into their own extension classes to shorten and cleanup Program.cs

**Detections**
- Tweaked style on Submit button to match spacing with other buttons
- Changed the component so that modifyable content is only available when a user is logged in as a moderator
- Changed the component so that the Submit button is only available when a user is logged in as a moderator

**Confirmed, False Positive, Unknown Detections**
- Changed the pages so that a detection's information (Tags, Comments, Results) can be updated after the initial moderator has reviewed

**Dashboard/User Activity**
- Changed so that clicking a detection link associated with a Tag opens it in a new tab so that
  users don't have to reload the tag list when they close the detection link

### 2021-03-15
#### AIForOrcas.Server
**DetectionsController**
- Added filter by hydrophone location to the GetAsync, GetUnreviewedAsync, GetConfirmedAsync, GetFalsePositivesAsync, and GetUnknownsAsync endpoints
- Added 3-hour and 6-hour options to the timeframe filter for the GetAsync, GetUnreviewedAsync, GetConfirmedAsync, GetFalsePositivesAsync, and GetUnknownsAsync endpoints

**MetricsController**
- Fixed bug related to capitalization differences and sorting problems in hand-entered Tags for GetSystemMetrics and GetModeratorMetrics endpoints

#### AIForOrcas.Client.BL
**DateHelper**
- Fixed bug that was causing DST dates to incorrectly convert

#### AIForOrcas.Client.Web
**CandidateFilterComponent (used by Candidates and Unknowns pages)**
- Added hydrophone location filter
- Added 3 hour and 6 hour options to the timespan filter

**ReviewedFilterComponent (used by Confirmed and False Positives pages)**
- Added hydrophone location filter
- Added 3 hour and 6 hour options to the timespan filter in the ReviewedFilterComponent 

**DetectionComponent**
- Fixed bug that kept the .WAV file playing even after the Spectrogram/Details modal is closed
- Fixed bug that was allowing drag and drop action on the Spectrogram/Details modal to incorrectly draw new regions
- Fixed bug that was allowing regions on the Spectrogram/Details modal to be resized

**Candidates page**
- Set the default timespan option to 6 hours 

## Archive
There are two additional folders in ModeratorFrontEnd folder and represent prototype/POC code. They should NOT be built, published or deployed to Azure as they will not work with the currently deployed CosmosDB instance.
