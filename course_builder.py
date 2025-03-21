def generate_course_structure(videos, difficulty="beginner"):
    """Organizes videos into a structured course."""
    course = {"title": f"Course on {videos[0]['title']}", "difficulty": difficulty, "modules": []}

    num_videos = len(videos)
    num_modules = min(5, num_videos // 3)  # Divide into modules (max 5)

    for i in range(num_modules):
        module_videos = videos[i * 3: (i + 1) * 3]  # 3 videos per module
        course["modules"].append({
            "title": f"Module {i+1}: {module_videos[0]['title']}",
            "videos": module_videos
        })

    return course
