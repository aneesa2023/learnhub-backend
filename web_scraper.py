import requests
from bs4 import BeautifulSoup


def scrape_article(url):
    """Extracts the main content from a webpage."""
    try:
        headers = {"User-Agent": "Mozilla/5.0"}
        response = requests.get(url, headers=headers, timeout=10)

        if response.status_code != 200:
            return None

        soup = BeautifulSoup(response.text, "html.parser")
        paragraphs = soup.find_all("p")

        text = " ".join([p.text for p in paragraphs[:5]])  # Limit to first 5 paragraphs
        return text.strip() if text else None

    except Exception as e:
        print(f"❌ Web Scraping Error: {str(e)}")
        return None
