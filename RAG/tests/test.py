import boto3

bedrock = boto3.client("bedrock", region_name="us-west-2")
print(bedrock.list_foundation_models())