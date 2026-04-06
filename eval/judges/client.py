import boto3

class BedrockClient:
    def __init__(self, model_id):
        self.client = boto3.client("bedrock-runtime")
        self.model_id = model_id

    def chat(self, messages, temperature=0.0, max_tokens=512):
        prompt = self._format(messages)

        response = self.client.converse(
            modelId=self.model_id,
            messages=[{"role": "user", "content": [{"text": prompt}]}],
            inferenceConfig={
                "maxTokens": max_tokens,
                "temperature": temperature
            }
        )

        return response["output"]["message"]["content"][0]["text"]

    def _format(self, messages):
        return "\n".join([m["content"] for m in messages])