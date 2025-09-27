import os
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel
from google_search_results import GoogleSearch 
import google.generativeai as genai
from dotenv import load_dotenv
from datetime import datetime

# Load environment variables
load_dotenv()

# Keys from .env
SERPAPI_KEY = os.getenv("SERPAPI_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not SERPAPI_KEY or not GEMINI_API_KEY:
    raise RuntimeError("Missing SERPAPI_KEY or GEMINI_API_KEY in environment variables")

# Configure Gemini
genai.configure(api_key=GEMINI_API_KEY)

# FastAPI app
app = FastAPI(title="City Events API", description="Fetch and structure upcoming events", version="1.0")


class Event(BaseModel):
    name: str
    date: str
    place: str
    booking_link: str
    description: str


@app.get("/events", response_model=list[Event])
async def get_events(city: str = Query(..., description="City name to search for upcoming events")):
    try:
        # Step 1: Fetch events using SerpApi
        search = GoogleSearch({
            "engine": "google_events",
            "q": f"upcoming events in {city}",
            "hl": "en",
            "api_key": SERPAPI_KEY
        })
        results = search.get_dict()

        if "events_results" not in results:
            raise HTTPException(status_code=404, detail="No events found")

        raw_events = results["events_results"]

        # Step 2: Send events to Gemini to structure
        prompt = f"""
        Today's date is {datetime.now().strftime("%Y-%m-%d")}.
        Here are some raw event details:
        {raw_events}

        Please extract and return them as a structured JSON array.
        Each event must have: name, date, place, booking_link, description.
        """

        model = genai.GenerativeModel("gemini-1.5-flash")
        response = model.generate_content(prompt)

        # Parse Gemini JSON response
        try:
            import json
            structured_events = json.loads(response.text)
        except Exception:
            raise HTTPException(status_code=500, detail="Failed to parse Gemini response into JSON")

        return structured_events

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
