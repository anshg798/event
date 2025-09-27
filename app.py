import os
import json
import requests
from typing import List
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from serpapi.google_search_results import GoogleSearch  # ✅ Correct PyPI import
from dotenv import load_dotenv
import google.generativeai as genai
from bs4 import BeautifulSoup

# --- Load environment variables ---
load_dotenv()
SERPAPI_KEY = os.getenv("SERPAPI_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not SERPAPI_KEY:
    raise ValueError("SERPAPI_KEY missing in .env file")
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)

# --- FastAPI setup ---
app = FastAPI(title="India Events API - Structured + Gemini")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Models ---
class EventRequest(BaseModel):
    city: str

class Event(BaseModel):
    date: str
    name: str
    place: str
    booking_link: str
    description: str

# --- Parse SerpApi results ---
def parse_serpapi_results(results: List[dict], city: str) -> List[dict]:
    events = []
    for r in results:
        title = r.get("title") or ""
        link = r.get("link") or ""
        snippet = r.get("snippet") or ""
        events.append({
            "date": "-",  # placeholder
            "name": title,
            "place": city,
            "booking_link": link,
            "description": snippet
        })
    return events

# --- Extract full HTML content from booking link ---
def fetch_full_description(link: str, fallback_description: str) -> str:
    try:
        resp = requests.get(link, timeout=10)
        if resp.status_code == 200:
            soup = BeautifulSoup(resp.text, "html.parser")
            text = soup.get_text(separator=" ", strip=True)
            return text if text else fallback_description
    except Exception:
        pass
    return fallback_description

# --- Extract structured events using Gemini ---
def extract_events_from_description(description: str, booking_link: str, city: str) -> List[dict]:
    if not GEMINI_API_KEY:
        return [{
            "date": "-",
            "name": description[:50] + "...",
            "place": city,
            "booking_link": booking_link,
            "description": description
        }]
    try:
        full_text = fetch_full_description(booking_link, description)
        prompt = (
            "You are an expert assistant. Extract all upcoming events from the following text. "
            "Each event must have: name, date, place, booking_link, description. "
            f"Booking link for all events: {booking_link}, city: {city}.\n\n"
            f"Text:\n{full_text}\n\n"
            "Return the output strictly as a JSON array of events."
        )
        model = genai.GenerativeModel("gemini-2.5-flash")
        resp = model.generate_content(
            prompt,
            generation_config={"temperature": 0, "max_output_tokens": 3500}
        )
        text = resp.text.strip()
        if text.startswith("```"):
            text = "\n".join(text.split("\n")[1:-1]).strip()
        events = json.loads(text)
        structured_events = []
        for e in events:
            structured_events.append({
                "name": e.get("name", "N/A"),
                "date": e.get("date", "N/A"),
                "place": e.get("place", city),
                "booking_link": e.get("booking_link", booking_link),
                "description": e.get("description", description)
            })
        return structured_events
    except Exception:
        return [{
            "date": "-",
            "name": description[:50] + "...",
            "place": city,
            "booking_link": booking_link,
            "description": description
        }]

# --- Normalize & deduplicate ---
def normalize_events(events: List[dict]) -> List[Event]:
    out = []
    seen = set()
    for e in events:
        key = (e.get("name", "").strip().lower(), e.get("date", ""))
        if key in seen:
            continue
        seen.add(key)
        out.append(Event(
            date=e.get("date", "") or "N/A",
            name=e.get("name", "") or "N/A",
            place=e.get("place", "") or "N/A",
            booking_link=e.get("booking_link", "") or "N/A",
            description=e.get("description", "") or "N/A"
        ))
    out.sort(key=lambda x: x.date)
    return out

# --- API endpoint ---
@app.post("/get-events", response_model=List[Event])
async def get_events(request: EventRequest):
    city = request.city
    query = f"upcoming events in {city} next 3 months site:in"

    try:
        search = GoogleSearch({
            "engine": "google",
            "q": query,
            "api_key": SERPAPI_KEY,
            "num": 20
        })
        results_json = search.get_dict()
        results = results_json.get("organic_results", [])
        initial_events = parse_serpapi_results(results, city)

        all_events = []
        for e in initial_events:
            extracted = extract_events_from_description(e["description"], e["booking_link"], city)
            all_events.extend(extracted)

        normalized = normalize_events(all_events)
        if not normalized:
            raise HTTPException(status_code=404, detail=f"No events found for {city}")
        return normalized
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch events: {str(e)}")

# --- Run server ---
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
