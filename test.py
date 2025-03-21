from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import json
import requests
import boto3
import os
from dotenv import load_dotenv

# ✅ Load environment variables from .env file
load_dotenv()

# ✅ AWS Credentials
AWS_REGION = os.getenv("AWS_REGION", "us-east-1")
AWS_ACCESS_KEY = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")

# ✅ Ensure AWS credentials exist
if not AWS_ACCESS_KEY or not AWS_SECRET_KEY:
    raise Exception("❌ AWS credentials are missing! Set AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY.")

# ✅ Initialize AWS Bedrock Client
bedrock_client = boto3.client(
    "bedrock-runtime",
    region_name=AWS_REGION,
    aws_access_key_id=AWS_ACCESS_KEY,
    aws_secret_access_key=AWS_SECRET_KEY
)

# ✅ YouTube API Key
YOUTUBE_API_KEY = os.getenv("YOUTUBE_API_KEY")

if not YOUTUBE_API_KEY:
    raise Exception("❌ YouTube API Key is missing! Set YOUTUBE_API_KEY in your .env file.")

# ✅ FastAPI App
app = FastAPI()

# ✅ Model IDs (Update this with valid ones from AWS CLI)
MODEL_IDS = {
    "Technical & Programming": "anthropic.claude-v2",
    "Mathematics and Algorithms": "anthropic.claude-v2",
    "Science & Engineering": "anthropic.claude-v2",
    "History & Social Studies": "anthropic.claude-3-sonnet-20240229-v1:0",  # Update if incorrect
    "Creative Writing & Literature": "anthropic.claude-3-sonnet-20240229-v1:0",
    "Business & Finance": "anthropic.claude-3-sonnet-20240229-v1:0",
    "Health & Medicine": "amazon.titan-text-express-v1",
}

# ✅ Course Request Model
class CourseRequest(BaseModel):
    topic: str
    description: str
    category: str
    difficulty: str
    chapters: int
    tone_output_style: str  # Combines Tone & Output Style


@app.post("/generate-course/")
def generate_course(request: CourseRequest):
    """
    Generates a structured course using AWS Bedrock Claude/Titan and fetches related YouTube videos.
    """
    try:
        print(f"🔹 Generating course for topic: {request.topic}, Category: {request.category}")

        # Step 1: Determine Model & Prompt
        model_id, prompt = get_model_and_prompt(request)

        if not model_id:
            raise HTTPException(status_code=500, detail=f"❌ No valid model found for category: {request.category}")

        print(f"🔹 Using Model: {model_id}")

        # Step 2: Call AWS Bedrock AI
        ai_response = call_bedrock_api(model_id, prompt)

        if not ai_response:
            raise HTTPException(status_code=500, detail="AI model failed to generate course.")

        # Step 3: Parse AI Response & Fetch YouTube Videos
        course_data = json.loads(ai_response)

        for chapter in course_data.get("chapters", []):
            chapter["videos"] = fetch_youtube_videos(chapter["search_queries"])

        # Step 4: Store & Return Course Data
        return {"message": "✅ Course generated successfully!", "course": course_data}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"❌ Error: {str(e)}")


def get_model_and_prompt(request):
    """
    Determines the AI model and selects the appropriate structured prompt based on category.
    """
    model_id = MODEL_IDS.get(request.category, None)

    if not model_id:
        print(f"⚠️ No model found for category: {request.category}. Defaulting to Claude 3 Sonnet.")
        model_id = "anthropic.claude-3-sonnet-20240229-v1:0"

    prompt = f"""
    Generate a structured course outline for "{request.topic}".

    Course Details:
    - Short Description: {request.description}
    - Category: {request.category}
    - Difficulty Level: {request.difficulty}
    - Number of Chapters: {request.chapters}
    - Tone/Output Style: {request.tone_output_style}

    Expected Output:
    {{
        "course_title": "Generated Course Title",
        "difficulty": "{request.difficulty}",
        "tone_output_style": "{request.tone_output_style}",
        "course_description": "Generated Course Description",
        "chapters": [
            {{
                "chapter_number": 1,
                "chapter_title": "Generated Chapter 1 Title",
                "chapter_description": "Summary of chapter content",
                "search_queries": [
                    "{request.topic} tutorial 2024",
                    "{request.topic} step by step guide",
                    "{request.topic} in-depth explanation"
                ]
            }},
            {{
                "chapter_number": 2,
                "chapter_title": "Generated Chapter 2 Title",
                "chapter_description": "Summary of chapter content",
                "search_queries": [
                    "Advanced {request.topic} tutorial",
                    "{request.topic} expert level guide",
                    "{request.topic} real-world applications"
                ]
            }}
        ]
    }}
    """

    return model_id, prompt


def call_bedrock_api(model_id, prompt):
    """
    Calls AWS Bedrock with the appropriate model and prompt.
    """
    try:
        print(f"🔹 Sending request to AWS Bedrock {model_id}...")

        payload = {
            "anthropic_version": "bedrock-2023-05-31",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt}
                    ]
                }
            ],
            "max_tokens": 800,
            "temperature": 0.7,
            "top_p": 0.9
        }

        response = bedrock_client.invoke_model(
            modelId=model_id,
            body=json.dumps(payload),
            contentType="application/json",
            accept="application/json"
        )

        response_body = json.loads(response["body"].read().decode("utf-8"))

        return response_body.get("content", [{}])[0].get("text", "No response generated.")

    except Exception as e:
        print(f"❌ AWS Bedrock API Error: {str(e)}")
        return None


def fetch_youtube_videos(search_queries):
    """
    Fetches relevant YouTube videos using the YouTube Data API.
    """
    try:
        videos = []
        for query in search_queries:
            url = f"https://www.googleapis.com/youtube/v3/search?part=snippet&q={query}&key={YOUTUBE_API_KEY}&maxResults=5&type=video"
            response = requests.get(url).json()

            for item in response.get("items", []):
                videos.append({
                    "video_title": item["snippet"]["title"],
                    "video_link": f"https://www.youtube.com/watch?v={item['id']['videoId']}",
                    "channel_name": item["snippet"]["channelTitle"]
                })

        return videos

    except Exception as e:
        print(f"❌ YouTube API Error: {str(e)}")
        return []


# ✅ Run FastAPI using:
# uvicorn main:app --reload
