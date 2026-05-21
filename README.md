# Smart Visual Electronics Search & Recommender

AI-powered electronics product search: upload a photo, get instant product identification, similar item matches, and expert recommendations.

## Architecture

```
User
 │
 ▼
Streamlit UI (port 8501)
 │  POST /search (image upload)
 ▼
FastAPI Backend (port 8000)
 ├── YOLOv8 → detect & crop product
 ├── CLIP ViT-B/32 → 512-dim embedding
 ├── Qdrant (port 6333) → vector similarity search
 └── Gemini 2.0 Flash → natural language recommendation
 │
 ▼
SearchResponse (JSON) → displayed in Streamlit
```

## Quick Start

**Prerequisites:** Docker & Docker Compose installed.

1. **Clone the repo**
   ```bash
   git clone <repo-url>
   cd smart-visual-search
   ```

2. **Set up environment variables**
   ```bash
   cp .env.example .env
   # Edit .env and add your Gemini API key
   # Get a free key at https://aistudio.google.com/app/apikey
   ```

3. **Start all services**
   ```bash
   docker-compose up --build
   ```
   This starts Qdrant, the FastAPI backend, and the Streamlit frontend.

4. **Ingest the product catalog** (run after services are up)
   ```bash
   # Option A: inside Docker
   docker-compose exec backend python /app/../scripts/ingest.py

   # Option B: locally (requires pip install of backend requirements)
   python scripts/ingest.py
   ```
   First time? Download the dataset first — see [Dataset Setup](#dataset-setup) below.

5. **Open the app**
   Navigate to [http://localhost:8501](http://localhost:8501)

## Dataset Setup

The system uses the [Amazon Sales Dataset](https://www.kaggle.com/datasets/karkavelrajaj/amazon-sales-dataset) from Kaggle.

**Automatic download (requires Kaggle API token):**
```bash
pip install kaggle pandas
# Place ~/.kaggle/kaggle.json (from https://www.kaggle.com/settings → Create New Token)
python scripts/download_dataset.py
```

**Manual download:**
1. Go to [https://www.kaggle.com/datasets/karkavelrajaj/amazon-sales-dataset](https://www.kaggle.com/datasets/karkavelrajaj/amazon-sales-dataset)
2. Download and extract the archive
3. Place `amazon.csv` in the `data/` directory
4. Run `python scripts/download_dataset.py` to filter and save `data/electronics_catalog.csv`
5. Then run `python scripts/ingest.py` to embed and load into Qdrant

## Model Choices

| Component | Choice | Reason |
|---|---|---|
| Object detection | YOLOv8n | Fast, accurate, pre-trained on COCO with electronics classes (laptop, cell phone, TV, etc.) |
| Image embedding | CLIP ViT-B/32 | Cross-modal semantic embeddings; strong zero-shot visual similarity for products |
| Vector database | Qdrant | Purpose-built for similarity search; persistent storage; cosine distance support |
| LLM | Gemini 2.0 Flash | Free tier via Google AI Studio; fast inference; strong structured output for comparisons |

## API Reference

**Health check**
```bash
curl http://localhost:8000/health
# {"status":"ok","qdrant":"connected"}
```

**Catalog stats**
```bash
curl http://localhost:8000/catalog/stats
# {"total_products":1842,"collection":"electronics_products"}
```

**Visual search**
```bash
curl -X POST http://localhost:8000/search \
  -F "image=@/path/to/laptop.jpg"
# Returns: SearchResponse JSON with detected_category, matches[], recommendation
```

**Interactive API docs:** [http://localhost:8000/docs](http://localhost:8000/docs)

## Known Limitations

- **Image quality sensitivity:** Blurry, dark, or cluttered images reduce YOLO detection confidence and CLIP embedding quality.
- **YOLO class coverage:** YOLO is trained on COCO classes — some electronics (e.g., earbuds, gaming controllers) may not be detected, falling back to the full image.
- **Catalog size:** Demo is capped at 2,000 products for speed. Larger catalogs require more ingestion time and RAM.
- **Gemini free tier rate limits:** The free API tier has RPM limits; heavy usage may trigger fallback messages.
- **Image URL availability:** Product images from the Amazon dataset may be unavailable (404/blocked), causing some catalog entries to be skipped during ingestion.
