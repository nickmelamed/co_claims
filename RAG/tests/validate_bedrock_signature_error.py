"""
Validate whether Bedrock auth is failing with IncompleteSignatureException.

Usage:
  python validate_bedrock_signature_error.py

Exit codes:
  0 -> Call succeeded (no signature error)
  2 -> Expected IncompleteSignatureException detected
  1 -> Any other failure
"""

import os
import sys

import boto3
from botocore.exceptions import BotoCoreError, ClientError
from dotenv import load_dotenv


EXPECTED_CODE = "IncompleteSignatureException"
EXPECTED_TEXT_SNIPPETS = [
    "Authorization header requires",
    "Credential",
    "Signature",
    "SignedHeaders",
    "X-Amz-Date",
]


def mask(value: str, keep: int = 4) -> str:
    if not value:
        return "<missing>"
    if len(value) <= keep:
        return "*" * len(value)
    return ("*" * (len(value) - keep)) + value[-keep:]


def get_region() -> str:
    return (
        os.getenv("AWS_REGION")
        or os.getenv("AWS_DEFAULT_REGION")
        or "us-east-1"
    )


def validate_incomplete_signature_error() -> int:
    load_dotenv()

    access_key = os.getenv("AWS_ACCESS_KEY_ID", "")
    secret_key = os.getenv("AWS_SECRET_ACCESS_KEY", "")
    session_token = os.getenv("AWS_SESSION_TOKEN", "")
    bearer_token = os.getenv("AWS_BEARER_TOKEN_BEDROCK", "")
    region = get_region()

    print("=== AWS Env Check (safe output) ===")
    print(f"AWS_ACCESS_KEY_ID: {mask(access_key)}")
    print(f"AWS_SECRET_ACCESS_KEY: {mask(secret_key)}")
    print(f"AWS_SESSION_TOKEN: {mask(session_token)}")
    print(f"AWS_BEARER_TOKEN_BEDROCK: {mask(bearer_token)}")
    print(f"Region: {region}")
    print()

    try:
        client = boto3.client("bedrock", region_name=region)
        result = client.list_foundation_models()
        model_count = len(result.get("modelSummaries", []))
        print(
            "SUCCESS: list_foundation_models call completed. "
            f"Found {model_count} model summaries."
        )
        return 0
    except ClientError as exc:
        error = exc.response.get("Error", {})
        code = error.get("Code", "")
        message = error.get("Message", str(exc))
        print(f"ClientError Code: {code}")
        print(f"ClientError Message: {message}")

        if code == EXPECTED_CODE and all(
            snippet in message for snippet in EXPECTED_TEXT_SNIPPETS
        ):
            print()
            print("VALIDATED: Matched expected IncompleteSignatureException.")
            return 2

        print()
        print("FAILED: ClientError occurred, but not the expected signature error.")
        return 1
    except BotoCoreError as exc:
        print(f"BotoCoreError: {exc}")
        return 1
    except Exception as exc:  # pylint: disable=broad-except
        print(f"Unexpected error: {exc}")
        return 1


if __name__ == "__main__":
    sys.exit(validate_incomplete_signature_error())
