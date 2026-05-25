from dotenv import load_dotenv
load_dotenv()
import os
import logging

import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
FALLBACK = "Recommendation unavailable. Here are your top visual matches above."


def build_prompt(detected_category: str, matches: list) -> str:
    lines = [
        f'You are an electronics product expert. A user has uploaded a photo of what appears to be a "{detected_category}".',
        "",
        "Based on visual similarity search, here are the closest matching products from our catalog:",
        "",
    ]
    for i, match in enumerate(matches, 1):
        p = match if isinstance(match, dict) else match.__dict__
        specs_text = "\n".join(f"    - {v}" for v in p.get("specs", {}).values())
        lines += [
            f"**Rank {i}: {p.get('name', 'N/A')}**",
            f"  Brand: {p.get('brand', 'N/A')}",
            f"  Category: {p.get('category', 'N/A')}",
            f"  Price: ₹{p.get('price', 0):,.0f}",
            f"  Rating: {p.get('rating', 0)}/5",
            f"  Similarity Score: {p.get('score', 0):.3f}",
            "  Key Specs:",
            specs_text if specs_text else "    - N/A",
            "",
        ]

    lines += [
        "Please provide:",
        "1. IDENTIFICATION: What product this most likely is (brand, model, category) based on the matches.",
        "2. COMPARISON TABLE: A markdown table comparing the top 3 matches across: Price, Key Specs, Rating, Best For.",
        "3. RECOMMENDATION: Which product you'd recommend and why, considering value for money.",
        "4. PRICE ASSESSMENT: Whether the prices seem fair for what's offered.",
        "",
        "Be concise, factual, and helpful. Format your response in clean markdown.",
    ]
    return "\n".join(lines)


def get_recommendation(prompt: str) -> str:
    try:
        genai.configure(api_key=GEMINI_API_KEY)
        model = genai.GenerativeModel("gemini-2.5-flash")
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        logger.warning(f"Gemini recommendation failed: {e}")
        return FALLBACK


if __name__ == "__main__":
    test_matches = [
        {
            "name": "Samsung Galaxy S23",
            "brand": "Samsung",
            "category": "Smartphones",
            "price": 74999,
            "rating": 4.5,
            "score": 0.92,
            "specs": {"spec_1": "6.1-inch display", "spec_2": "8GB RAM", "spec_3": "128GB storage"},
        }
    ]
    prompt = build_prompt("cell phone", test_matches)
    print("--- Prompt ---")
    print(prompt)
    print("\n--- Recommendation ---")
    result = get_recommendation(prompt)
    print(result)
