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
This is a standard .NET Core Web API found in the AIForOrcas.Server project and is published to Azure as https://aifororcasdetections.azurewebsites.net

#### Publish Settings
- Site Url: https://aifororcasdetections.azurewebsites.net
- Resource group: AIForOrcas
- Configuration: Release
- Target framework: netcoreapp3.1
- Deployment mode: Self-contained
- Target runtime: win-x86

### A Web Fontend
This is a .NET Core Blazor-based web site found in the AIForOrcas.Client.Web project and is published to Azure as https://aifororcas.azurewebsites.net

#### Publish Settings
- Site Url: https://aifororcas.azurewebsites.net
- Resource group: AIForOrcas
- Configuration: Release
- Target framework: netcoreapp3.1
- Deployment mode: Self-contained
- Target runtime: win-x86

### Additional Projects
The remaining projects in the solution represent code that is shared between the two projects (i.e. DTOs/Models, Business Logic, etc.)

## Release Notes

### 2021-03-15
#### AIForOrcas.Server
**DetectionsController**
- Added filter by hydrophone location to the GetAsync, GetUnreviewedAsync, GetConfirmedAsync, GetFalsePositivesAsync, and GetUnknownsAsync endpoints
- Added 3-hour and 6-hour options to the timeframe filter for the GetAsync, GetUnreviewedAsync, GetConfirmedAsync, GetFalsePositivesAsync, and GetUnknownsAsync endpoints

**MetricsController**
- Fixed bug related to capitalization differences and sorting problems in hand-entered Tags for GetSystemMetrics and GetModeratorMetrics endpoints
- Fixed bug related to sorting error for 

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