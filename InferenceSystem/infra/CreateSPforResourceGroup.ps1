./Write-Log.ps1
<#
[AI4Orca l Hack Project]
AI4Orca Hack Project resource provisioning
1. Create Service Principal for GitHub Actions
  
#>


# Pre-request:  Azure ML CLI : az extension add --name azure-cli-ml

Write-Host  "==== AI4Orca Hack Project -  Create Service Principal Services  ===="
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

# Create the service principal with rights scoped to the subscription.
#$SP_PASSWD=az ad sp create-for-rbac --name $config.SERVICE_PRINCIPAL_NAME --scopes --scopes /subscriptions/$($config.AzureSubscriptionId)/resourceGroups/$($config.ResourceGroupName) --role contributor --sdk-auth --query password --output tsv
#$SP_PASSWD=az ad sp create-for-rbac --name $config.SERVICE_PRINCIPAL_NAME --scopes --scopes /subscriptions/$($config.AzureSubscriptionId)/resourceGroups/$($config.ResourceGroupName) --role contributor --sdk-auth --query password --output tsv
#$SP_PASSWD=az ad sp create-for-rbac --name $config.SERVICE_PRINCIPAL_NAME --scopes $ACR_REGISTRY_ID --role contributor --query password --output tsv
$SP=az ad sp create-for-rbac --name $config.SERVICE_PRINCIPAL_NAME --scopes $rg.id --role Contributor --sdk-auth | ConvertFrom-Json

$ACR_REGISTRY_ID=az acr show --name $config.ACR.ACRName --query id --output tsv

# Output the service principal's credentials; use these in your services and
# applications to authenticate to the container registry.
Write-Host "Service principal ID: $($SP.clientId)"
Write-Host "Service principal password: $($SP.clientSecret)"

# Store Value to KeyVault
az keyvault secret set --vault-name $config.AzureKeyVaultName --name "$($config.SERVICE_PRINCIPAL_NAME)-id" --value $SP.clientId
az keyvault secret set --vault-name $config.AzureKeyVaultName --name "$($config.SERVICE_PRINCIPAL_NAME)-pwd" --value $SP.clientSecret





