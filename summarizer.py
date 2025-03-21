import boto3
import json
import os
from dotenv import load_dotenv

# ✅ Load AWS credentials
load_dotenv()

AWS_REGION = os.getenv("AWS_REGION", "us-east-1")
AWS_ACCESS_KEY = os.getenv("AWS_ACCESS_KEY")
AWS_SECRET_KEY = os.getenv("AWS_SECRET_KEY")
USE_CLAUDE_3 = os.getenv("USE_CLAUDE_3_5", "False").lower() == "true"
MODEL_ID = os.getenv("MODEL_ID", "amazon.titan-text-express-v1")
INFERENCE_PROFILE_ARN = os.getenv("INFERENCE_PROFILE_ARN")

# ✅ Initialize AWS Bedrock Client
bedrock_client = boto3.client(
    "bedrock-runtime",
    region_name=AWS_REGION,
    aws_access_key_id=AWS_ACCESS_KEY,
    aws_secret_access_key=AWS_SECRET_KEY
)

def summarize_text(text):
    """Summarizes text using AWS Bedrock AI Model."""
    try:
        print(f"🔹 Sending request to AWS Bedrock... Using Model: {MODEL_ID}")

        if USE_CLAUDE_3:
            if not INFERENCE_PROFILE_ARN:
                raise ValueError("❌ ERROR: INFERENCE_PROFILE_ARN is missing for Claude 3!")

            payload = {
                "messages": [
                    {"role": "user", "content": f"Summarize the following text concisely:\n{text}"}
                ],
                "max_tokens": 300,
                "temperature": 0.7,
                "top_p": 0.9
            }
            response = bedrock_client.invoke_model(
                modelId=MODEL_ID,
                inferenceProfileArn=INFERENCE_PROFILE_ARN,
                body=json.dumps(payload),
                contentType="application/json",
                accept="application/json"
            )
        else:
            payload = {
                "inputText": f"Summarize this text: {text}",
                "textGenerationConfig": {
                    "maxTokenCount": 300,
                    "temperature": 0.7,
                    "topP": 0.9
                }
            }
            response = bedrock_client.invoke_model(
                modelId=MODEL_ID,
                body=json.dumps(payload),
                contentType="application/json",
                accept="application/json"
            )

        # ✅ Debugging response
        result = json.loads(response["body"].read().decode("utf-8"))
        print("✅ Bedrock Full Response:", result)

        # ✅ Extract summary from response
        summary = result.get("outputText", "No summary generated.").strip()

        return summary

    except Exception as e:
        print(f"❌ AWS Bedrock Summarization Error: {str(e)}")
        return f"Summarization failed. Error: {str(e)}"
