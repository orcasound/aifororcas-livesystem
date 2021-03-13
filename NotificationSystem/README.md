## AI For Orcas - Notification System

The notification system is a set of azure functions responsible for:
- Facilitating adding/removing moderators and subscribers
- Identifying changes in the database and sending alerts

## Architecture

### Update email list

![add email architecture](img/add-email.png)

There are two Azure Functions that update the email list.

- ModeratorEmail is a REST API that writes to the email list
- SenderEmail is a REST API that writes to the email list
- Email list is implemented using Azure Tables, using either "Moderator" or "Subscriber" as the partition key

#### Sample REST calls

Add email to subscribers list:

```bash
curl -X POST -d '{"email": "sample@email.com"}' '<SubscriberEmailEndpoint>'
```

Delete email from subscribers list:

```bash
curl -X POST -d '{"email": "sample@email.com"}' '<SubscriberEmailEndpoint>'
```

Add email to moderators list:

```bash
curl -X POST -d '{"email": "sample@email.com"}' '<ModeratorEmailEndpoint>'
```

Delete email from moderators list:

```bash
curl -X POST -d '{"email": "sample@email.com"}' '<ModeratorEmailEndpoint>'
```

### Send email to moderators and subscribers

![send email architecture](img/send-email.png)

There are three other Azure Functions that make up the email notification system.

In the moderators flow:

- A change in the Cosmos DB metadata store triggers the SendModeratorEmail function
- If there is a newly detected orca call that requires a moderator to validate, the function fetches the relevant email list
- The function then calls SendGrid to send emails to moderators

In the subscribers flow:

- A change in the Cosmos DB metadata store triggers the DbToQueue function
- If there is a new orca call that the moderator has validated, the function sends a message to a queue
- The SendSubscriberEmail function periodically checks the queue
- If there are items in the queue, the function fetches the relevant email list
- The function then calls SendGrid to send emails to subscribers

## Prerequisites

- Access to the Orca Conservancy Azure subscription
- Install the [.NET Core 3.1 SDK](https://dotnet.microsoft.com/download/dotnet-core/3.1)
- Azure Function Tools
    - If using Visual Studio, include "Azure development" workload in installation
    - If using Visual Studio Code, add the "Azure Functions" extension
    - If using CLI, install the [Azure Functions Core Tools](https://docs.microsoft.com/en-us/azure/azure-functions/functions-run-local?tabs=linux%2Ccsharp%2Cbash#v2)
- If running locally - [Azure storage emulator](https://docs.microsoft.com/en-us/azure/storage/common/storage-use-emulator)

## Build 
To build the functions locally:

1. Go to /NotificationSystem directory (if not already)
2. If building from the command line, run 
    ```
    dotnet build NotificationSystem.csproj
    ```
3. If building from visual studio, simply open .csproj and build as normal

## Azure Resource Dependencies
All resources are located in resource group **LiveSRKWNotificationSystem**.

1. Storage account with queues, email template images and moderator/subscriber list: orcanotificationstorage
2. Metadata store (from which some functions are triggered): aifororcasmetadatastore
3. Azure function app: orcanotification
4. SendGrid account (for sending emails): aifororcas

## Run Locally
It is recommended to go to the "orcanotification" function app, then Settings > Configuration to find the app settings used. 

1. Create a local.settings.json file with the settings found in the function app. See [documentation](https://docs.microsoft.com/en-us/azure/azure-functions/functions-run-local?tabs=windows%2Ccsharp%2Cbash#local-settings-file) here.
    - Set all connection strings to be "UseDevelopmentStorage=true"
2. You also need to start the azure storage emulator if you aren't using real connection strings. 

## Run on Azure

1. Go to the "orcanotification" function app (link 3 above). 
2. On the "Overview" tab, make sure the status of the function shows running.
3. On the "Functions" tab, you should see all the functions of the notification system. Enable/Disable as needed.