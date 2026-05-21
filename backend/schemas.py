from pydantic import BaseModel


class ProductMatch(BaseModel):
    id: str
    name: str
    brand: str
    category: str
    price: float
    rating: float
    specs: dict[str, str]
    image_url: str
    score: float


class SearchResponse(BaseModel):
    detected_category: str
    detected_confidence: float
    matches: list[ProductMatch]
    recommendation: str
    query_embedding_dim: int
