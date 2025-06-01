from azure.ai.ml import MLClient
from azure.identity import DefaultAzureCredential

ml_client = MLClient.from_config(credential=DefaultAzureCredential())
data_asset = ml_client.data.mount("azureml:solar-segmentation-data:initial",
                                  mount_point="/mnt/cache/solar-segmentation-data")
print(f"Data asset mounted at: {data_asset.mount_point}")
