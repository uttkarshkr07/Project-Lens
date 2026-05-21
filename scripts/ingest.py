import io
import logging
import os
import re
import sys
import uuid

import numpy as np
import pandas as pd
import requests
from PIL import Image
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams
from dotenv import load_dotenv

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "backend"))
from pipeline import embed_image

load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", "6333"))
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "electronics_products")
DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
CATALOG_CSV = os.path.join(DATA_DIR, "electronics_catalog.csv")
MAX_ROWS = 2000
BATCH_SIZE = 50
VECTOR_SIZE = 512


def parse_price(price_str: str) -> float:
    cleaned = re.sub(r"[₹,\s]", "", str(price_str))
    try:
        return float(cleaned)
    except ValueError:
        return 0.0


def parse_specs(about: str) -> dict:
    if not isinstance(about, str) or not about.strip():
        return {}
    parts = re.split(r"[|.]", about)
    specs = {}
    for i, part in enumerate(parts[:5], 1):
        part = part.strip()
        if part:
            specs[f"spec_{i}"] = part
    return specs


def download_image(url: str) -> Image.Image | None:
    try:
        resp = requests.get(url, timeout=5, headers={"User-Agent": "Mozilla/5.0"})
        resp.raise_for_status()
        return Image.open(io.BytesIO(resp.content)).convert("RGB")
    except Exception as e:
        logger.warning(f"Failed to download image from {url}: {e}")
        return None


def ensure_collection(client: QdrantClient):
    existing = {c.name for c in client.get_collections().collections}
    if COLLECTION_NAME not in existing:
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE),
        )
        logger.info(f"Created collection '{COLLECTION_NAME}'")
    else:
        logger.info(f"Collection '{COLLECTION_NAME}' already exists")


def load_catalog() -> pd.DataFrame:
    if not os.path.exists(CATALOG_CSV):
        logger.error(f"Catalog not found at {CATALOG_CSV}. Run scripts/download_dataset.py first.")
        sys.exit(1)

    df = pd.read_csv(CATALOG_CSV, on_bad_lines="skip")
    df = df.dropna(subset=["product_name", "img_link", "discounted_price"])
    df["price"] = df["discounted_price"].apply(parse_price)
    df = df[df["price"] > 0]
    df = df.head(MAX_ROWS).reset_index(drop=True)
    logger.info(f"Loaded {len(df)} products for ingestion (capped at {MAX_ROWS})")
    return df


def main():
    client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
    ensure_collection(client)

    df = load_catalog()
    total = len(df)
    batch: list[PointStruct] = []
    ingested = 0

    for idx, row in df.iterrows():
        img = download_image(str(row["img_link"]))
        if img is None:
            continue

        try:
            embedding = embed_image(img)
        except Exception as e:
            logger.warning(f"Embedding failed for row {idx}: {e}")
            continue

        brand = str(row.get("product_name", "")).split()[0] if str(row.get("product_name", "")).split() else "Unknown"
        payload = {
            "product_id": str(row.get("product_id", "")),
            "name": str(row.get("product_name", "")),
            "brand": brand,
            "category": str(row.get("category", "")),
            "price": float(row["price"]),
            "rating": float(str(row.get("rating", "0")).replace(",", ".")) if str(row.get("rating", "0")).replace(",", ".").replace(".", "", 1).isdigit() else 0.0,
            "specs": parse_specs(str(row.get("about_product", ""))),
            "image_url": str(row.get("img_link", "")),
        }

        batch.append(PointStruct(id=str(uuid.uuid4()), vector=embedding.tolist(), payload=payload))
        ingested += 1

        if len(batch) >= BATCH_SIZE:
            client.upsert(collection_name=COLLECTION_NAME, points=batch)
            batch.clear()
            print(f"Ingested {ingested}/{total} products...")

    if batch:
        client.upsert(collection_name=COLLECTION_NAME, points=batch)

    print(f"\nIngestion complete. Total successfully ingested: {ingested}/{total} products.")


if __name__ == "__main__":
    main()
