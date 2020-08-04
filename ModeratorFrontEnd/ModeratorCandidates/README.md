# Moderator Candidates API
2020-08-04: This API now ties in with live Cosmos DB system and represents the prelimiary data set for the end-to-end functionality. Suspect changes will be needed as the system continues to evolve.

2020-07-21: This is a preliminary attempt to design the REST endpoints needed to provide the Moderator Candidates 
front-end with appropriate data and functionality.

### How to use
You should be able to just run this solution in Visual Studio and it should automatically start up and provide the API.
If you are running it locally, you will need to add the cosmos db account key to appsetings.json. It is already set up for the
deployed version as a variable.

### How to deploy to Azure
The ModeratorCandidates.API project is configured to publish to the ModeratorCandidates App Service in Azure. You should be
able to publish your latest bits from Visual Studio via FTP using the FTP App credential set up in the Azure portal. To
retrieve the credential, go to the Azure portal and select App Services -> from the list selest moderatorcandidates -> in 
the ModeratorCandidates blade, click the Deployment Center under Deployment section -> under Manual Deployment (push / sync)
select FTP and then Dashboard -> on the Dashboard is the App Credentials for the FTP endpoint


### Known Issues

1. There are currently no authentication/authorization mechanisms in place. If you have the cosmos DB access key, you can access the data.
2. Tags are being reported back to Cosmos DB as a semi-colon separated string. They should  probably be stored as an array of some sort to improve sorting/indexing.
3. Internal exceptions are not being well handled at this point. See MetadataRepository.cs Commit() for example 