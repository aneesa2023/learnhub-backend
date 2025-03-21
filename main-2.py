import os
import requests
import json
import boto3
from fastapi import FastAPI, Query
from dotenv import load_dotenv

# ✅ Load API keys from .env
load_dotenv()
YOUTUBE_API_KEY = os.getenv("YOUTUBE_API_KEY")
AWS_REGION = os.getenv("AWS_REGION", "us-east-1")
AWS_ACCESS_KEY = os.getenv("AWS_ACCESS_KEY")
AWS_SECRET_KEY = os.getenv("AWS_SECRET_KEY")

# ✅ Initialize AWS Bedrock Client
bedrock_client = boto3.client(
    "bedrock-runtime",
    region_name=AWS_REGION,
    aws_access_key_id=AWS_ACCESS_KEY,
    aws_secret_access_key=AWS_SECRET_KEY
)

app = FastAPI(title="LearnHub Backend", description="AI-powered course generation API")

YOUTUBE_SEARCH_URL = "https://www.googleapis.com/youtube/v3/search"


def fetch_youtube_videos(topic):
    """Fetch relevant YouTube videos for the given topic."""
    params = {
        "part": "snippet",
        "q": topic,
        "key": YOUTUBE_API_KEY,
        "type": "video",
        "maxResults": 5
    }
    response = requests.get(YOUTUBE_SEARCH_URL, params=params)
    data = response.json()

    videos = []
    for item in data.get("items", []):
        video_data = {
            "title": item["snippet"]["title"],
            "url": f"https://www.youtube.com/watch?v={item['id']['videoId']}"
        }
        videos.append(video_data)
    return videos

def generate_course_outline(topic, description, level):
    """Use AWS Bedrock Claude 3 AI to generate a structured course outline based on level."""
    prompt = f"""
    You are an AI course creator. Create a structured course on "{topic}".
    - Course Level: {level} (Beginner/Intermediate/Advanced)
    - Short Description: {description}
    - Generate 3-5 modules with meaningful titles.
    - Structure the course based on the level.
    - Format the response as JSON:
      {{
        "title": "Course Name",
        "modules": [
          {{"title": "Module 1 Title"}},
          {{"title": "Module 2 Title"}},
          {{"title": "Module 3 Title"}}
        ]
      }}
    """

    response = bedrock_client.invoke_model(
        modelId="us.anthropic.claude-3-5-sonnet-20241022-v2:0",
        body=json.dumps({"prompt": prompt, "max_tokens_to_sample": 500}),
        contentType="application/json",
        accept="application/json"
    )

    result = json.loads(response["body"].read().decode("utf-8"))
    return json.loads(result.get("completion", "{}"))  # Ensure output is JSON


@app.get("/generate-course/")
def generate_course(
        topic: str = Query(..., title="Course Topic"),
        description: str = Query(..., title="Short Course Description"),
        level: str = Query(..., title="Difficulty Level", enum=["Beginner", "Intermediate", "Advanced"])
):
    """Generate a structured course using AWS Bedrock & YouTube API based on level & description."""
    course_outline = generate_course_outline(topic, description, level)
    course_modules = course_outline.get("modules", [])

    for module in course_modules:
        module["videos"] = fetch_youtube_videos(module["title"])  # Fetch relevant videos for each module

    return {
        "title": course_outline.get("title", f"Course on {topic}"),
        "description": description,
        "level": level,
        "modules": course_modules
    }
