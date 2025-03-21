import requests
import os
from dotenv import load_dotenv

# ✅ Load API Key from .env
load_dotenv()
YOUTUBE_API_KEY = os.getenv("YOUTUBE_API_KEY")

def search_youtube_videos(query, max_results=10):
    """Fetches relevant YouTube videos based on a topic."""
    url = f"https://www.googleapis.com/youtube/v3/search"
    params = {
        "part": "snippet",
        "q": query,
        "type": "video",
        "maxResults": max_results,
        "key": YOUTUBE_API_KEY
    }
    response = requests.get(url, params=params)
    data = response.json()

    videos = []
    for item in data.get("items", []):
        video_id = item["id"]["videoId"]
        title = item["snippet"]["title"]
        description = item["snippet"]["description"]
        published_at = item["snippet"]["publishedAt"]
        channel_title = item["snippet"]["channelTitle"]
        thumbnail_url = item["snippet"]["thumbnails"]["high"]["url"]

        videos.append({
            "video_id": video_id,
            "title": title,
            "description": description,
            "published_at": published_at,
            "channel": channel_title,
            "thumbnail": thumbnail_url,
            "url": f"https://www.youtube.com/watch?v={video_id}"
        })

    return videos
