# Moderator Candidates API
2020-07-21: This is a preliminary attempt to design the REST endpoints needed to provide the Moderator Candidates 
front-end with appropriate data and functionality.

### How to use
You should be able to just run this solution in Visual Studio and it should automatically start up and provide the API.

### Hot to add data
Right now, the .json files containing the metadata are store in the solution's Data folder, any additions or should be made here and will
be reflected in the API the next time it is started.

### What it is not
The .json files are static and don't represent live content.  That is expected to be provided by some repository or other content provider
in the future and the appropriate service (in the Services folder) will need to be updated to access the live content (i.e SQL, Cosmos DB, etc.)