from fastapi import FastAPI, HTTPException
from fastapi.middleware import Middleware
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from enum import Enum
from typing import List, Dict, Any, Optional
import json
import requests
import boto3
import os
import time
from functools import wraps
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Constants
YOUTUBE_LIMITS = {
    "videos_per_keyword": 1,
    "keywords_per_chapter": 5,
    "max_total_videos": 3
}

MODEL_IDS = {
    "Technical & Programming": "us.anthropic.claude-3-5-sonnet-20241022-v2:0",
    "Mathematics and Algorithms": "us.anthropic.claude-3-5-sonnet-20241022-v2:0",
    "Science & Engineering": "anthropic.claude-3-sonnet-20240229-v1:0",
    "History & Social Studies": "anthropic.claude-3-sonnet-20240229-v1:0",
    "Creative Writing & Literature": "anthropic.claude-3-sonnet-20240229-v1:0",
    "Business & Finance": "anthropic.claude-3-sonnet-20240229-v1:0",
    "Health & Medicine": "amazon.titan-text-express-v1",
    "General": "us.anthropic.claude-3-5-sonnet-20241022-v2:0"
}

# AWS Configuration
AWS_REGION = os.getenv("AWS_REGION", "us-east-1")
AWS_ACCESS_KEY = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")
YOUTUBE_API_KEY = os.getenv("YOUTUBE_API_KEY")

if not all([AWS_ACCESS_KEY, AWS_SECRET_KEY, YOUTUBE_API_KEY]):
    raise Exception("Missing required environment variables")

bedrock_client = boto3.client(
    "bedrock-runtime",
    region_name=AWS_REGION,
    aws_access_key_id=AWS_ACCESS_KEY,
    aws_secret_access_key=AWS_SECRET_KEY
)

# Middleware
middleware = [
    Middleware(
        CORSMiddleware,
        allow_origins=["http://localhost:3000"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
]

app = FastAPI(title="Learning Path Generator API", middleware=middleware)

# Enums and Models
class CourseCategory(str, Enum):
    technical = "Technical & Programming"
    math = "Mathematics and Algorithms"
    science = "Science & Engineering"
    history = "History & Social Studies"
    literature = "Creative Writing & Literature"
    business = "Business & Finance"
    health = "Health & Medicine"
    general = "General"

class DifficultyLevel(str, Enum):
    beginner = "Beginner"
    intermediate = "Intermediate"
    advanced = "Advanced"

class OutputStyle(str, Enum):
    educational = "Educational"
    conversational = "Conversational"
    formal = "Formal"
    storytelling = "Storytelling"

class Video(BaseModel):
    video_title: str
    video_id: str
    video_link: str
    channel_name: str
    description: str
    thumbnail: str
    publish_date: str
    search_query: str

class YouTubeResources(BaseModel):
    total_videos: int
    videos: List[Video]
    limits_applied: Dict[str, Any]

class KeyConcept(BaseModel):
    title: str
    explanation: str

class ChapterContent(BaseModel):
    chapter_number: int
    chapter_title: str
    learning_objectives: List[str]
    key_concepts: List[KeyConcept]
    practical_applications: List[str]
    study_notes: str
    youtube_keywords: List[str]
    videos: Optional[YouTubeResources] = None

class LearningPathSummary(BaseModel):
    overview: str
    time_commitment: str
    assessment_methods: List[str]
    next_steps: List[str]

class CourseRequest(BaseModel):
    topic: str
    description: str
    category: CourseCategory
    difficulty: DifficultyLevel
    chapters: int
    tone_output_style: OutputStyle

class LearningPathResponse(BaseModel):
    course_title: str
    difficulty: str
    description: str
    chapters: List[ChapterContent]
    learning_path_summary: LearningPathSummary
    metadata: Dict[str, Any]

# Utils
def rate_limit(calls_per_second=1):
    min_interval = 1.0 / calls_per_second
    last_call_time = {}

    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            current_time = time.time()
            last_time = last_call_time.get(func.__name__, 0)
            sleep_time = min_interval - (current_time - last_time)
            if sleep_time > 0:
                time.sleep(sleep_time)
            result = await func(*args, **kwargs)
            last_call_time[func.__name__] = time.time()
            return result

        return wrapper
    return decorator

def call_bedrock_api(model_id: str, prompt: str) -> str:
    payload = {
        "anthropic_version": "bedrock-2023-05-31",
        "messages": [{"role": "user", "content": [{"type": "text", "text": prompt}]}],
        "max_tokens": 4096,
        "temperature": 0.5,
        "top_p": 0.9
    }
    response = bedrock_client.invoke_model(
        modelId=model_id,
        body=json.dumps(payload),
        contentType="application/json",
        accept="application/json"
    )
    response_body = json.loads(response["body"].read().decode("utf-8"))
    return response_body.get("content", [{}])[0].get("text", "")

def fetch_youtube_videos(search_queries: List[str]) -> Dict[str, Any]:
    videos, total_videos = [], 0
    for query in search_queries[:YOUTUBE_LIMITS["keywords_per_chapter"]]:
        if total_videos >= YOUTUBE_LIMITS["max_total_videos"]:
            break
        url = (
            f"https://www.googleapis.com/youtube/v3/search?part=snippet"
            f"&q={query}&key={YOUTUBE_API_KEY}"
            f"&maxResults={YOUTUBE_LIMITS['videos_per_keyword']}"
            f"&type=video&relevanceLanguage=en&videoEmbeddable=true"
        )
        response = requests.get(url)
        if response.status_code != 200:
            continue
        for item in response.json().get("items", []):
            if total_videos >= YOUTUBE_LIMITS["max_total_videos"]:
                break
            video_data = {
                "video_title": item["snippet"]["title"],
                "video_id": item["id"]["videoId"],
                "video_link": f"https://www.youtube.com/watch?v={item['id']['videoId']}",
                "channel_name": item["snippet"]["channelTitle"],
                "description": item["snippet"]["description"],
                "thumbnail": item["snippet"]["thumbnails"]["medium"]["url"],
                "publish_date": item["snippet"]["publishTime"],
                "search_query": query
            }
            if not any(v["video_id"] == video_data["video_id"] for v in videos):
                videos.append(video_data)
                total_videos += 1
    return {
        "total_videos": len(videos),
        "videos": videos,
        "limits_applied": YOUTUBE_LIMITS
    }

def construct_learning_path_prompt(request: CourseRequest) -> str:
    return f'''
Please create a JSON-formatted learning path for the topic "{request.topic}" with the following inputs:
- Description: {request.description}
- Category: {request.category}
- Difficulty: {request.difficulty}
- Chapters: {request.chapters}
- Tone/Style: {request.tone_output_style}

Output must strictly follow this JSON schema:
{{
  "course_title": "string",
  "description": "string",
  "chapters": [
    {{
      "chapter_number": int,
      "chapter_title": "string",
      "learning_objectives": ["string"],
      "key_concepts": [{{"title": "string", "explanation": "string"}}],
      "practical_applications": ["string"],
      "study_notes": "string (at least 2000 characters)",
      "youtube_keywords": ["string"]
    }}
  ],
  "learning_path_summary": {{
    "overview": "string",
    "time_commitment": "string",
    "assessment_methods": ["string"],
    "next_steps": ["string"]
  }}
}}
Return only the JSON. No markdown, no explanation.
'''

def fix_ai_chapter_format(chapter, index):
    chapter["chapter_number"] = index + 1
    if "key_concepts" in chapter:
        chapter["key_concepts"] = [
            {"title": c.get("title", c.get("concept", "")), "explanation": c.get("explanation", "")}
            for c in chapter["key_concepts"]
        ]
    return ChapterContent(**chapter)

@rate_limit()
async def fetch_youtube_content(keywords: List[str]) -> YouTubeResources:
    return YouTubeResources(**fetch_youtube_videos(keywords))

@app.post("/generate-learning-path/", response_model=LearningPathResponse)
async def generate_learning_path(request: CourseRequest):
    try:
        model_id = MODEL_IDS.get(request.category, MODEL_IDS["General"])
        prompt = construct_learning_path_prompt(request)
        ai_response = call_bedrock_api(model_id, prompt)
        print("\n🧠 AI Raw Response:", ai_response)
        course_data = json.loads(ai_response)

        enhanced_chapters = []
        for i, ch in enumerate(course_data["chapters"]):
            chapter_model = fix_ai_chapter_format(ch, i)
            yt_data = await fetch_youtube_content(chapter_model.youtube_keywords)
            chapter_model.videos = yt_data
            enhanced_chapters.append(chapter_model)

        return LearningPathResponse(
            course_title=course_data["course_title"],
            difficulty=request.difficulty,
            description=request.description,
            chapters=enhanced_chapters,
            learning_path_summary=LearningPathSummary(**course_data["learning_path_summary"]),
            metadata={
                "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                "youtube_resources_count": sum(len(c.videos.videos) for c in enhanced_chapters),
                "total_chapters": len(enhanced_chapters)
            }
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail={"error": str(e)})
