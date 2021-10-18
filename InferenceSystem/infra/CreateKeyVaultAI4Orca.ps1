./Write-Log.ps1
<#
[Orca Call ML Project]
Orca Call ML Project resource provisioning
1. Including the provisioning the following resources:
   Azure ML:
#>


# Pre-request:  Azure ML CLI : az extension add --name azure-cli-ml

Write-Host  "==== AI for Orca Project -  Create KeyVault    ===="
Write-Host   "======================================================" 

#Get Global Config
$config = Get-Content .\provision-config.json | ConvertFrom-Json

#Start to deploy Azure Resources
Write-Host  "Before deployment, please make sure you have installed the Powershell Core (version 6.x up) and latest Azure Cli"

az login
az account set --subscription $config.AzureSubscriptionId

$rg=az group show -n $config.ResourceGroupName | ConvertFrom-Json

# Check Resource Group exists
if($rg -eq $null)
{
    Write-Output ("Resource Group $($config.ResourceGroupName) not exists")
    Write-Output ("Creating/Assign Resource Group for Deployment")
    az group create -n $config.ResourceGroupName -l $config.Location
}
else
{
    Write-Output ("Resource Group $($rg.name) exists")
}

# Create Azure keyvault
az keyvault create --name $config.AzureKeyVaultName --resource-group $config.ResourceGroupName -l $config.Location

# Get SignIn Object ID
$loginuserobj=az ad signed-in-user show | ConvertFrom-Json
Write-Output ("Login User ID  $($loginuserobj.objectId) exists")

# Set Azure KeyValut Policy
az keyvault set-policy --name $config.AzureKeyVaultName --object-id $loginuserobj.objectId --secret-permissions delete get list purge  set --key-permissions create decrypt delete encrypt get import list purge  sign unwrapKey update verify wrapKey --certificate-permissions  create delete deleteissuers  get getissuers import list listissuers managecontacts manageissuers purge setissuers update
