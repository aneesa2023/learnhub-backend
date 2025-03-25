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
from datetime import datetime
from fastapi import Body, Request
from fastapi.responses import JSONResponse
from fastapi import APIRouter
import re

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

middleware = [
    Middleware(
        CORSMiddleware,
        allow_origins=["http://localhost:3000"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
]

app = FastAPI(title="LearnHub API", middleware=middleware)

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
    view_count: Optional[int] = 0
    like_count: Optional[int] = 0
    score: Optional[float] = 0.0
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
    recommended_study_links: Optional[List[str]] = []
    course_summary: Optional[str] = None

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

def safe_json_loads(response_text: str):
    try:
        response_text = re.sub(r"[\x00-\x1f\x7f]", "", response_text)
        return json.loads(response_text)
    except json.JSONDecodeError as e:
        raise HTTPException(status_code=500, detail={
            "error": "Claude returned invalid JSON",
            "json_error": str(e),
            "raw_output": response_text
        })

def call_bedrock_api(model_id: str, prompt: str) -> str:
    payload = {
        "anthropic_version": "bedrock-2023-05-31",
        "messages": [{"role": "user", "content": [{"type": "text", "text": prompt}]}],
        "max_tokens": 4096,
        "temperature": 0.5,
        "top_p": 0.9
    }
    for attempt in range(5):
        try:
            response = bedrock_client.invoke_model(
                modelId=model_id,
                body=json.dumps(payload),
                contentType="application/json",
                accept="application/json"
            )
            response_body = json.loads(response["body"].read().decode("utf-8"))
            return response_body.get("content", [{}])[0].get("text", "")
        except bedrock_client.exceptions.ThrottlingException:
            wait = 2 ** attempt
            print(f"🕐 Throttled. Retrying in {wait} seconds...")
            time.sleep(wait)
        except Exception as e:
            raise HTTPException(status_code=500, detail={"error": str(e)})
    raise HTTPException(status_code=429, detail={"error": "Too many requests to Claude. Please try again later."})

def fetch_youtube_videos(search_queries: List[str]) -> Dict[str, Any]:
    videos = []
    for query in search_queries[:YOUTUBE_LIMITS["keywords_per_chapter"]]:
        search_url = (
            f"https://www.googleapis.com/youtube/v3/search?part=snippet"
            f"&q={query}"
            f"&key={YOUTUBE_API_KEY}"
            f"&maxResults=10"
            f"&type=video"
            f"&order=relevance"  
            # f"&videoDuration=medium"  # Or "long" for deep content
            f"&safeSearch=strict"
        )
        search_response = requests.get(search_url)
        if search_response.status_code != 200:
            continue

        search_items = search_response.json().get("items", [])
        video_ids = [item["id"]["videoId"] for item in search_items if "videoId" in item.get("id", {})]

        if video_ids:
            video_ids_str = ",".join(video_ids)
            stats_url = (
                f"https://www.googleapis.com/youtube/v3/videos?part=snippet,statistics"
                f"&id={video_ids_str}&key={YOUTUBE_API_KEY}"
            )
            stats_response = requests.get(stats_url)
            if stats_response.status_code != 200:
                continue

            for item in stats_response.json().get("items", []):
                try:
                    views = int(item["statistics"].get("viewCount", 0))
                    likes = int(item["statistics"].get("likeCount", 0))
                    published_at = item["snippet"]["publishedAt"]
                    pub_date = datetime.strptime(published_at, "%Y-%m-%dT%H:%M:%SZ")
                    # days_old = max(1, (datetime.utcnow() - pub_date).days)
                    # score = (likes * 2 + views) / days_old

                    video_data = {
                        "video_title": item["snippet"]["title"],
                        "video_id": item["id"],
                        "video_link": f"https://www.youtube.com/watch?v={item['id']}",
                        "channel_name": item["snippet"]["channelTitle"],
                        "description": item["snippet"].get("description", ""),
                        "thumbnail": item["snippet"]["thumbnails"]["medium"]["url"],
                        "publish_date": published_at,
                        "view_count": views,
                        "like_count": likes,
                        # "score": score,
                        "search_query": query
                    }

                    if not any(v["video_id"] == video_data["video_id"] for v in videos):
                        videos.append(video_data)
                except Exception:
                    continue

    videos.sort(key=lambda v: v.get("score", 0), reverse=True)
    top_videos = videos[:YOUTUBE_LIMITS["max_total_videos"]]

    return {
        "total_videos": len(top_videos),
        "videos": top_videos,
        "limits_applied": YOUTUBE_LIMITS
    }

def build_course_summary_prompt(course_title: str, chapters: List[Dict[str, Any]]) -> str:
    chapter_titles = ", ".join([f'"{c["chapter_title"]}"' for c in chapters])
    return f"""
Write a 200-word summary for a course titled "{course_title}". 
The course covers the following chapters: {chapter_titles}.
The summary should introduce the course, highlight key learning outcomes, 
and end with what learners can expect to achieve.
Format: plain text only. No markdown or JSON.
"""

def generate_prompt(category, topic, description, difficulty, chapters, tone_output_style):
    base_instructions = f"""
Topic: {topic}
Difficulty: {difficulty}
Total Chapters: {chapters}
Tone: {tone_output_style}
Format: STRICT STRUCTURED JSON
Description: {description}

Generate ONLY the introduction and chapter outline in the format below:
{{
  "course_title": "string",
  "description": "string",
  "chapters": [
    {{
      "chapter_number": int,
      "chapter_title": "string",
      "summary": "4-5 sentence overview of the chapter"
    }}
  ]
}}

Do NOT generate full chapter content yet. Avoid markdown or commentary.
"""

    enhancements = {
        "Technical & Programming": "\nInclude code examples in study_notes. Each chapter should teach core programming concepts. Generate YouTube keywords like 'How to use React Hooks' or 'Python classes with real-world examples'.",
        "Mathematics and Algorithms": "\nInclude math formulas and derivations. Explain concepts with step-by-step solutions. YouTube keywords should be like 'Dynamic Programming tutorial', 'Graph theory walkthrough', or 'Mathematical induction with examples'.",
        "Science & Engineering": "\nIncorporate scientific concepts, definitions, and diagrams. Use real-life experiments where applicable. YouTube keywords like 'Ohm’s Law experiment', 'Thermodynamics animation', or 'Engineering statics example'.",
        "History & Social Studies": "\nCover important historical events, timelines, and figures. Include causes, consequences, and quotes. YouTube keywords like 'World War 2 summary', 'French Revolution explained', or 'Renaissance in Europe documentary'.",
        "Creative Writing & Literature": "\nFocus on literary techniques, plot development, and author styles. Include short writing prompts or poem examples. YouTube keywords like 'Shakespeare sonnet breakdown', 'Creative writing prompts', or 'Literary devices explained'.",
        "Business & Finance": "\nInclude business models, case studies, frameworks (like SWOT, PESTLE), and real data examples. YouTube keywords like 'How stock markets work', 'Business model canvas', or 'Finance basics for beginners'.",
        "Health & Medicine": "\nFocus on structured medical topics: definitions, symptoms, causes, treatments. Include labeled diagrams if relevant. YouTube keywords like 'Cardiovascular system explained', 'Mental health awareness', or 'Human anatomy 3D'.",
        "General": "\nEnsure clarity and engagement. Use metaphors or real-world analogies. Keep tone simple and helpful. YouTube keywords like 'Easy guide to [topic]', 'Explained like I’m 5', or 'Intro to [topic] animation'."
    }

    category_notes = enhancements.get(category, enhancements["General"])
    return base_instructions + category_notes

def build_intro_prompt(req: CourseRequest) -> str:
    return generate_prompt(
        category=req.category,
        topic=req.topic,
        description=req.description,
        difficulty=req.difficulty,
        chapters=req.chapters,
        tone_output_style=req.tone_output_style
    )

def build_chapter_prompt(req: CourseRequest, intro: str, chapter_title: str, chapter_number: int) -> str:
    return f"""
You are continuing from this course introduction:

{intro}

Now write full content for Chapter {chapter_number}: \"{chapter_title}\" for the course \"{req.topic}\".

🎯 Guidelines for `study_notes` field:
- Use short paragraphs and bullet points for clarity.
- Include visual section headers like "🔥 Key Insights", "💡 Examples", "🧠 Summary".
- Add inline code snippets using backticks for technical terms or examples.
- Highlight important definitions and concepts in **bold**.
- Where applicable, provide visual metaphors or real-world analogies.
- Add code examples, formulas, or sample problems if relevant.
- Maintain an audio-friendly, engaging tone.

⚠️ Output format rules:
- Return response in **strict JSON** only, no markdown or commentary.
- `study_notes` must be a single string (not markdown), with at least 2000 characters of rich, structured content.
- No introductory or closing statements like “In conclusion” or “Thank you”.

Format must be:

{{
  "chapter_number": int,
  "chapter_title": "string",
  "learning_objectives": ["string"],
  "key_concepts": [{{"title": "string", "explanation": "string"}}],
  "practical_applications": ["string"],
  "study_notes": "Minimum 2000 characters of structured, audio-friendly, formatted explanations.",
  "youtube_keywords": ["string"]
}}
"""

def fix_ai_chapter_format(chapter, index):
    chapter["chapter_number"] = index + 1
    if "key_concepts" in chapter:
        chapter["key_concepts"] = [
            {"title": c.get("title", c.get("concept", "")), "explanation": c.get("explanation", "")}
            for c in chapter["key_concepts"]
        ]
    return ChapterContent(**chapter)

@app.post("/generate-learning-path/", response_model=LearningPathResponse)
async def generate_learning_path(request: CourseRequest):
    try:
        model_id = MODEL_IDS.get(request.category, MODEL_IDS["General"])

        intro_prompt = build_intro_prompt(request)
        intro_response = call_bedrock_api(model_id, intro_prompt)
        print("🧪 Claude Raw Intro Response:\n", intro_response)  # 👈 Add this

        try:
            intro_data = json.loads(intro_response)
        except json.JSONDecodeError as decode_err:
            raise HTTPException(status_code=500, detail={
                "error": "Claude returned invalid JSON",
                "json_error": str(decode_err),
                "raw_output": intro_response
            })

        intro_text = f"{intro_data['description']}\n\nChapters:\n" + "\n".join(
            [f"{c['chapter_number']}. {c['chapter_title']}" for c in intro_data['chapters']]
        )

        full_chapters = []
        for chapter in intro_data["chapters"]:
            chapter_prompt = build_chapter_prompt(
                request,
                intro_text,
                chapter_title=chapter["chapter_title"],
                chapter_number=chapter["chapter_number"]
            )
            chapter_response = call_bedrock_api(model_id, chapter_prompt)
            # chapter_data = json.loads(chapter_response)
            chapter_data = safe_json_loads(chapter_response)
            chapter_model = fix_ai_chapter_format(chapter_data, chapter["chapter_number"] - 1)
            yt_data = fetch_youtube_videos(chapter_model.youtube_keywords)
            chapter_model.videos = YouTubeResources(**yt_data)
            full_chapters.append(chapter_model)

        # 🧠 Add global recommended study links from top video links
        study_links = []
        for chapter in full_chapters:
            if chapter.videos and chapter.videos.videos:
                for video in chapter.videos.videos:
                    if video.video_link not in study_links:
                        study_links.append(video.video_link)

        # Just keep top 3–5 links
        study_links = study_links[:5]
        summary_prompt = build_course_summary_prompt(
            intro_data["course_title"],
            intro_data["chapters"]
        )
        summary_text = call_bedrock_api(model_id, summary_prompt).strip()

        # Prepare final course object
        course_data = {
            "course_title": intro_data["course_title"],
            "difficulty": request.difficulty,
            "description": request.description,
            "chapters": [c.dict() for c in full_chapters],
            "learning_path_summary": {
                "overview": "This course provides a deep dive into the topic with practical chapters and visual resources.",
                "time_commitment": "Approx. 1–2 weeks",
                "assessment_methods": ["Quizzes", "Mini Projects", "Discussions"],
                "next_steps": ["Explore advanced topics", "Join communities", "Apply knowledge"],
                "recommended_study_links": study_links,
                "course_summary": summary_text
            },
            "metadata": {
                "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                "youtube_resources_count": sum(len(c.videos.videos) for c in full_chapters),
                "total_chapters": len(full_chapters)
            }
        }

        # Upload to S3
        filename = f"{intro_data['course_title'].replace(' ', '_')}_{int(time.time())}"
        s3_uri = upload_course_to_s3(course_data, filename)

        print(f"✅ Course uploaded to S3: {s3_uri}")

        # Return course (without s3_uri in model but you can add it to metadata)
        return LearningPathResponse(
            course_title=course_data["course_title"],
            difficulty=course_data["difficulty"],
            description=course_data["description"],
            chapters=full_chapters,
            learning_path_summary=LearningPathSummary(**course_data["learning_path_summary"]),
            metadata={
                **course_data["metadata"],
                "s3_uri": s3_uri
            }
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail={"error": str(e)})

@app.post("/upload-course-to-s3")
def upload_course_to_s3(course_data: dict, filename: str) -> str:
    s3 = boto3.client(
        "s3",
        region_name=AWS_REGION,
        aws_access_key_id=AWS_ACCESS_KEY,
        aws_secret_access_key=AWS_SECRET_KEY
    )

    bucket_name = os.getenv("S3_BUCKET_NAME")
    folder = os.getenv("S3_FOLDER", "courses")
    key = f"{folder}/{filename}.json"

    s3.put_object(
        Bucket=bucket_name,
        Key=key,
        Body=json.dumps(course_data, indent=2),
        ContentType="application/json"
    )

    return f"s3://{bucket_name}/{key}"

from fastapi.responses import JSONResponse

@app.get("/list-courses/")
def list_courses():
    try:
        s3 = boto3.client(
            "s3",
            region_name=AWS_REGION,
            aws_access_key_id=AWS_ACCESS_KEY,
            aws_secret_access_key=AWS_SECRET_KEY
        )
        bucket_name = os.getenv("S3_BUCKET_NAME")
        folder = os.getenv("S3_FOLDER", "courses")

        result = s3.list_objects_v2(Bucket=bucket_name, Prefix=f"{folder}/")

        files = [
            obj["Key"].split("/")[-1].replace(".json", "")
            for obj in result.get("Contents", [])
            if obj["Key"].endswith(".json")
        ]
        return {"courses": files}
    except Exception as e:
        raise HTTPException(status_code=500, detail={"error": str(e)})

@app.get("/get-course/{course_name}")
def get_course(course_name: str):
    try:
        s3 = boto3.client(
            "s3",
            region_name=AWS_REGION,
            aws_access_key_id=AWS_ACCESS_KEY,
            aws_secret_access_key=AWS_SECRET_KEY
        )
        bucket_name = os.getenv("S3_BUCKET_NAME")
        folder = os.getenv("S3_FOLDER", "courses")
        key = f"{folder}/{course_name}.json"

        response = s3.get_object(Bucket=bucket_name, Key=key)
        course_data = json.loads(response["Body"].read())
        return JSONResponse(content=course_data)
    except Exception as e:
        raise HTTPException(status_code=500, detail={"error": str(e)})

