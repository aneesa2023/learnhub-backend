# LearnHub Backend 🚀  
**Welcome to LearnHub** – AI-Powered Personalized Learning Paths  
Generate structured, interactive learning paths with AI-generated chapters, curated YouTube videos, and rich study notes – all stored securely on AWS.

---

## 🔧 What It Does

- 🎯 User inputs a **topic**, **description**, **category**, **difficulty**, and **tone**
- 📚 AI (Claude via Amazon Bedrock) generates a complete **course** with:
  - Chapter titles
  - Learning objectives
  - Key concepts
  - Study notes
  - Practical applications
- 📹 Top **YouTube videos** are fetched and ranked using view counts, likes, and recency
- 📦 Final course is **uploaded to Amazon S3** as a structured JSON
- 🔄 Course is retrievable by ID for rendering on frontend (Explore, Detail pages)

---

## 🛠 Tech Stack

- **Backend**: FastAPI (Python 3.9+)
- **AI**: Amazon Bedrock (Claude 3.5, Titan)
- **Video Search**: YouTube Data API
- **Storage**: Amazon S3
- **Auth-Ready**: Amazon Cognito compatible
- **Serverless Support**: Mangum for AWS Lambda

---

## ⚙️ Endpoints

### `POST /generate-learning-path/`
- Accepts course input and returns full structured content
- Includes AI-generated chapters, video links, metadata

### `POST /upload-course-to-s3`
- Uploads a given course to S3
- Returns the `s3_uri` and generated `course_id`

### `GET /get-course/{course_id}`
- Fetches course from S3 using the ID

### `GET /list-courses/`
- Lists all saved courses in the S3 folder

---

## 🚀 Getting Started

### 1. Clone the repo

```bash
git clone https://github.com/your-username/learnhub-backend.git
cd learnhub-backend
---

### 2. Create .env file

AWS_REGION=us-east-1
AWS_ACCESS_KEY_ID=your-access-key
AWS_SECRET_ACCESS_KEY=your-secret-key
S3_BUCKET_NAME=your-s3-bucket
S3_FOLDER=courses
YOUTUBE_API_KEY=your-youtube-api-key

3. Install dependencies
pip install -r requirements.txt

4. Run the server
bash
Copy
Edit
uvicorn main:app --reload

📦 Sample Request
POST /generate-learning-path/
{
  "topic": "Web Accessibility",
  "description": "Learn how to build inclusive web apps",
  "category": "Technical & Programming",
  "difficulty": "Beginner",
  "chapters": 3,
  "tone_output_style": "Educational"
}

🧠 Architecture Highlights
💡 Claude Sonnet 3.5 is used for structured JSON course content

🔁 Retry & throttling logic for Bedrock and YouTube APIs

🧠 YouTube curation uses search + stats API to select top videos

📤 S3 Upload stores each course as courses/{filename}.json

⚡ Mangum used for serverless (AWS Lambda) compatibility

✅ Features Checklist
 Claude prompt-based chapter generation

 YouTube video auto-curation

 Dynamic metadata (chapters, views, timestamps)

 Upload + fetch from S3

 Plug-and-play backend for any frontend

 OpenAPI/Swagger docs enabled by default

💡 Sample Use Cases
🧪 College students wanting bite-sized, YouTube-supported study plans

🧑‍🏫 Teachers generating quick custom curriculum

🎧 Audio learners preparing voice-based lessons (EchoPod integration)

🧩 Plugin for LMS systems to auto-generate paths

🧩 Next Features
 PDF export for full learning path

 Generate quizzes per chapter

 Visual knowledge graphs from chapters

 Course ratings + user feedback collection

 Instructor/Org multi-user support

📁 Project Structure

learnhub-backend/
│
├── main.py               # FastAPI application
├── .env                  # Secrets and keys
├── prompts/              # Prompt building logic
├── models/               # Pydantic schema definitions
├── utils/                # S3, Bedrock, YouTube helpers
├── requirements.txt      # Python dependencies
└── README.md
