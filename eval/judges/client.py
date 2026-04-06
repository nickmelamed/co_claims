import boto3

import os
import boto3

_bedrock_client = None


def get_bedrock_client():
    global _bedrock_client

    if _bedrock_client is None:
        _bedrock_client = boto3.client(
            "bedrock-runtime",
            region_name=os.getenv("AWS_REGION", "us-east-1")
        )

    return _bedrock_client


class BedrockClient:
    def __init__(self, model_id):
        self.model_id = model_id
        self.client = get_bedrock_client()

    def chat(self, prompt: str, temperature=0.0, max_tokens=512):
        response = self.client.converse(
            modelId=self.model_id,
            messages=[{
                "role": "user",
                "content": prompt   # string not list 
            }],
            inferenceConfig={
                "maxTokens": max_tokens,
                "temperature": temperature
            }
        )

        return response["output"]["message"]["content"][0]["text"]