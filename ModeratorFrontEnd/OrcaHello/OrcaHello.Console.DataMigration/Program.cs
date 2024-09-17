using Microsoft.Extensions.Configuration;
using OrcaHello.Console.DataMigration.Services;

class Program
{
    static void Main(string[] args)
    {
        var builder = new ConfigurationBuilder()
            .AddJsonFile("AppSettings.json", optional: true, reloadOnChange: true)
            .AddUserSecrets<Program>();

        IConfigurationRoot configuration = builder.Build();

        var service = new DatabaseMigrationService(configuration);

        while (true)
        {
            Console.Clear();

            Console.WriteLine("Please select an option:");
            Console.WriteLine("1. Show configuration values");
            Console.WriteLine("2. Copy database from online Cosmos DB to local emulator");
            Console.WriteLine("3. Create container on local emulator with new schema");
            Console.WriteLine("4. Create composite indexes on new container on local emulator");
            Console.WriteLine("5. Copy new container to online Cosmos DB");
            Console.WriteLine("10. Exit");

            var input = Console.ReadLine();

            switch (input)
            {
                case "1":
                    service.ShowConfiguration();
                    break;
                case "2":
                    service.CopyDatabaseFromOnlineToLocal().GetAwaiter().GetResult();
                    break;
                case "3":
                    service.CreateNewSchemaContainerOnLocal().GetAwaiter().GetResult();
                    break;
                case "4":
                    service.CreateCompositeIndexOnLocal().GetAwaiter().GetResult();
                    break;
                case "5":
                    service.CopyContainerToOnline().GetAwaiter().GetResult();
                    break;
                case "10":
                    return;
                default:
                    Console.WriteLine("Invalid option selected");
                    break;
            }
        }
    }
}