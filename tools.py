"""
Career Advisor Bot — LangChain Tools
  1. estimate_salary   — average salary for a role via Adzuna API.
  2. fetch_recent_jobs — most recent job openings via Adzuna API.
  3. course_recommender - recommended online courses via Youtube API.
"""

import os

import requests
from langchain.tools import tool


# Adzuna country codes for supported European locations

ADZUNA_COUNTRIES = {
    "germany":     "de",
    "netherlands": "nl",
    "france":      "fr",
    "spain":       "es",
    "poland":      "pl",
    "austria":     "at",
    "italy":       "it",
    "belgium":     "be",
}


# Tool 1: Salary Estimator (live data via Adzuna API)

@tool
def estimate_salary(role: str, location: str) -> str:
    """
    Return the average salary for a data science or technology role in a European
    country, using live Adzuna job market data.

    Args:
        role: The job title (e.g., 'Data Scientist', 'ML Engineer', 'Data Analyst').
        location: European country (e.g., 'Germany', 'Netherlands', 'France', 'Spain').
    """
    country_code = ADZUNA_COUNTRIES.get(location.lower().strip())
    if not country_code:
        available = ", ".join(k.title() for k in ADZUNA_COUNTRIES)
        return f"Location '{location}' not supported. Available: {available}."

    try:
        response = requests.get(
            f"https://api.adzuna.com/v1/api/jobs/{country_code}/histogram",
            params={"app_id": os.getenv("ADZUNA_APP_ID"), "app_key": os.getenv("ADZUNA_APP_KEY"), "what": role},
            timeout=10,
        )
        response.raise_for_status()
        histogram = response.json().get("histogram", {})
    except Exception as exc:
        return f"Could not retrieve salary data: {exc}"

    if not histogram or sum(histogram.values()) < 5:
        return f"Not enough data for '{role}' in {location.title()}. Try a broader title."

    total_count = sum(histogram.values())
    weighted_sum = sum((int(bucket) + 5_000) * count for bucket, count in histogram.items())
    average = int(weighted_sum / total_count)
    currency = "£" if country_code == "gb" else "€"

    return (
        f"Average salary for **{role.title()}** in **{location.title()}**: "
        f"**{currency}{average:,}/year**\n"
        f"*(Based on {total_count} live Adzuna postings)*"
    )


# Tool 2: Recent Job Openings (live data via Adzuna API)

@tool
def fetch_recent_jobs(role: str, location: str) -> str:
    """
    Fetch the 5 most recent job openings for a role in a European country
    using live Adzuna job market data.

    Args:
        role: The job title (e.g., 'Data Scientist', 'ML Engineer', 'Data Analyst').
        location: European country (e.g., 'Germany', 'Netherlands', 'France', 'Spain').
    """
    country_code = ADZUNA_COUNTRIES.get(location.lower().strip())
    if not country_code:
        available = ", ".join(k.title() for k in ADZUNA_COUNTRIES)
        return f"Location '{location}' not supported. Available: {available}."

    try:
        response = requests.get(
            f"https://api.adzuna.com/v1/api/jobs/{country_code}/search/1",
            params={
                "app_id": os.getenv("ADZUNA_APP_ID"),
                "app_key": os.getenv("ADZUNA_APP_KEY"),
                "what": role,
                "results_per_page": 5,
                "sort_by": "date",
            },
            timeout=10,
        )
        response.raise_for_status()
        results = response.json().get("results", [])
    except Exception as exc:
        return f"Could not fetch job listings: {exc}"

    if not results:
        return f"No recent openings found for '{role}' in {location.title()}."

    lines = [f"**Most recent {role.title()} openings in {location.title()}:**\n"]
    for i, job in enumerate(results, 1):
        title = job.get("title", "N/A")
        company = job.get("company", {}).get("display_name", "Unknown company")
        posted = job.get("created", "")[:10]
        url = job.get("redirect_url", "")
        lines.append(f"{i}. **{title}** — {company}")
        if posted:
            lines.append(f"   Posted: {posted}")
        if url:
            lines.append(f"   {url}")
        lines.append("")

    return "\n".join(lines)

# Tool 3: Course Recommender

@tool
def course_recommender(skill: str) -> str:
    """
    Recommend top YouTube tutorial videos for a given skill relevant to data science
    and technology careers, using the YouTube Data API.

    Args:
        skill: The skill to find tutorials for (e.g., 'Python', 'Machine Learning', 'SQL').
    """
    try:
        response = requests.get(
            "https://www.googleapis.com/youtube/v3/search",
            params={"key": os.getenv("YOUTUBE_API_KEY"), "q": f"{skill} tutorial",
                    "part": "snippet", "type": "video", "maxResults": 5},
            timeout=10,
        )
        response.raise_for_status()
        items = response.json().get("items", [])[:5]
    except Exception as e:
        return f"Could not fetch videos: {e}"

    lines = [f"**Top YouTube tutorials for '{skill}':**\n"]
    for i, item in enumerate(items, 1):
        title = item["snippet"]["title"]
        channel = item["snippet"]["channelTitle"]
        url = f"https://www.youtube.com/watch?v={item['id']['videoId']}"
        lines.append(f"{i}. **{title}** — {channel}\n   {url}\n")

    return "\n".join(lines)
