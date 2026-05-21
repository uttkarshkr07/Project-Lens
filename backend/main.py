import io
import os
import time
import logging

from fastapi import FastAPI, File, UploadFile, HTTPException, Response
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
from qdrant_client import QdrantClient
from dotenv import load_dotenv

from schemas import ProductMatch, SearchResponse
from pipeline import run_pipeline
from llm import build_prompt, get_recommendation

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", "6333"))
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "electronics_products")

app = FastAPI(title="Smart Visual Electronics Search", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

ALLOWED_CONTENT_TYPES = {"image/jpeg", "image/png", "image/jpg"}


@app.get("/health")
async def health():
    try:
        client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
        client.get_collections()
        return {"status": "ok", "qdrant": "connected"}
    except Exception as e:
        logger.error(f"Qdrant health check failed: {e}")
        from fastapi.responses import JSONResponse
        return JSONResponse(
            status_code=503,
            content={"status": "error", "qdrant": "unreachable"},
        )


@app.get("/catalog/stats")
async def catalog_stats():
    try:
        client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
        info = client.get_collection(COLLECTION_NAME)
        return {
            "total_products": info.points_count,
            "collection": COLLECTION_NAME,
        }
    except Exception as e:
        logger.error(f"Catalog stats failed: {e}")
        raise HTTPException(status_code=503, detail=f"Could not retrieve catalog stats: {e}")


@app.post("/search", response_model=SearchResponse)
async def search(response: Response, image: UploadFile = File(...)):
    if image.content_type not in ALLOWED_CONTENT_TYPES:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file type '{image.content_type}'. Only JPEG and PNG images are accepted.",
        )

    start = time.time()

    image_bytes = await image.read()
    try:
        pil_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Could not decode image: {e}")

    pipeline_result = run_pipeline(pil_image)

    matches_raw = pipeline_result["matches"]
    matches = []
    for hit in matches_raw:
        payload = hit.payload or {}
        matches.append(ProductMatch(
            id=str(hit.id),
            name=payload.get("name", "Unknown"),
            brand=payload.get("brand", "Unknown"),
            category=payload.get("category", "Unknown"),
            price=float(payload.get("price", 0.0)),
            rating=float(payload.get("rating", 0.0)),
            specs=payload.get("specs", {}),
            image_url=payload.get("image_url", ""),
            score=float(hit.score),
        ))

    prompt = build_prompt(pipeline_result["detected_category"], [m.model_dump() for m in matches])
    recommendation = get_recommendation(prompt)

    elapsed = time.time() - start
    response.headers["X-Processing-Time"] = f"{elapsed:.2f}s"

    return SearchResponse(
        detected_category=pipeline_result["detected_category"],
        detected_confidence=pipeline_result["detected_confidence"],
        matches=matches,
        recommendation=recommendation,
        query_embedding_dim=pipeline_result["embedding_dim"],
    )
