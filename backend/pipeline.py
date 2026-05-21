import os
import sys
import time
import logging

import numpy as np
import torch
import open_clip
from PIL import Image
from ultralytics import YOLO
from qdrant_client import QdrantClient
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", "6333"))
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "electronics_products")
TOP_K = int(os.getenv("TOP_K", "5"))

ELECTRONICS_CLASSES = ["cell phone", "laptop", "keyboard", "mouse", "remote", "tv", "clock"]

yolo_model = YOLO("yolov8n.pt")

clip_model, _, clip_preprocess = open_clip.create_model_and_transforms(
    "ViT-B-32", pretrained="openai"
)
clip_model.eval()


def detect_and_crop(image: Image.Image) -> tuple[Image.Image, str, float]:
    results = yolo_model(image, verbose=False)
    best_box = None
    best_conf = 0.0
    best_class = "unknown"

    for result in results:
        if result.boxes is None:
            continue
        for box in result.boxes:
            conf = float(box.conf[0])
            cls_name = result.names[int(box.cls[0])]
            if conf > best_conf:
                best_conf = conf
                best_box = box.xyxy[0].tolist()
                best_class = cls_name

    if best_box is None or best_conf < 0.3:
        return image, "unknown", 0.0

    x1, y1, x2, y2 = best_box
    pad = 10
    w, h = image.size
    x1 = max(0, x1 - pad)
    y1 = max(0, y1 - pad)
    x2 = min(w, x2 + pad)
    y2 = min(h, y2 + pad)
    cropped = image.crop((x1, y1, x2, y2))
    return cropped, best_class, best_conf


def embed_image(image: Image.Image) -> np.ndarray:
    tensor = clip_preprocess(image).unsqueeze(0)
    with torch.no_grad():
        embedding = clip_model.encode_image(tensor)
    embedding = embedding.squeeze(0).numpy().astype(np.float32)
    norm = np.linalg.norm(embedding)
    if norm > 0:
        embedding = embedding / norm
    return embedding


def _get_qdrant_client() -> QdrantClient:
    for attempt in range(10):
        try:
            client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
            client.get_collections()
            return client
        except Exception as e:
            if attempt < 9:
                logger.warning(f"Qdrant not ready (attempt {attempt + 1}/10): {e}. Retrying in 2s...")
                time.sleep(2)
            else:
                raise RuntimeError(f"Could not connect to Qdrant after 10 attempts: {e}")


def search_catalog(embedding: np.ndarray, top_k: int = TOP_K):
    client = _get_qdrant_client()
    results = client.search(
        collection_name=COLLECTION_NAME,
        query_vector=embedding.tolist(),
        limit=top_k,
        with_payload=True,
    )
    return results


def run_pipeline(image: Image.Image) -> dict:
    cropped_image, detected_category, detected_confidence = detect_and_crop(image)
    embedding = embed_image(cropped_image)
    matches = search_catalog(embedding)
    return {
        "cropped_image": cropped_image,
        "detected_category": detected_category,
        "detected_confidence": detected_confidence,
        "matches": matches,
        "embedding_dim": len(embedding),
    }


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python pipeline.py <image_path>")
        sys.exit(1)
    img_path = sys.argv[1]
    img = Image.open(img_path).convert("RGB")
    cropped, category, confidence = detect_and_crop(img)
    print(f"Detected: {category} (confidence: {confidence:.2f})")
    emb = embed_image(cropped)
    print(f"Embedding shape: {emb.shape}, dtype: {emb.dtype}")
    print(f"Embedding norm: {np.linalg.norm(emb):.4f}")
