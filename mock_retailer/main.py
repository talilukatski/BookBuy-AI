from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uuid
import pandas as pd
import os
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

app = FastAPI(title="Mock Retailer API")

CATALOG_DIR = os.path.join(os.path.dirname(__file__), "catalogs")
CATALOGS = {}


def load_catalogs():
    """Load CSV catalogs into memory as dicts for O(1) exact-title lookup."""
    global CATALOGS
    shops = ["fiction_boutique", "knowledge_store", "mega_market1", "mega_market2"]

    print("Loading catalogs...")
    loaded = {}

    for shop in shops:
        file_path = os.path.join(CATALOG_DIR, f"{shop}.csv")

        if not os.path.exists(file_path):
            logger.warning(f"  - Warning: {file_path} not found.")
            continue

        try:
            df = pd.read_csv(file_path)

            catalog = {}
            for _, row in df.iterrows():
                title = str(row["Title"]).strip()
                if not title:
                    continue

                key = title.lower()
                catalog[key] = {
                    "Title": title,
                    "price": float(row["price"]),
                    "stock": bool(row["stock"]),
                    "categories": str(row["categories"]) if "categories" in row and pd.notna(row["categories"]) else "Unknown",
                }

            loaded[shop] = catalog
            logger.info(f"  - Loaded {shop} ({len(catalog)} items)")

        except Exception as e:
            logger.error(f"  - Error loading {shop}: {e}")

    CATALOGS = loaded


load_catalogs()


class BuyRequest(BaseModel):
    title: str
    user_address: str
    payment_token: str


class BuyResponse(BaseModel):
    transaction_id: str
    status: str
    eta: str


class SearchResponse(BaseModel):
    title: str
    price: float
    stock: bool
    shop_id: str
    category: str


@app.get("/shops/{shop_id}/search", response_model=SearchResponse)
async def search_book(shop_id: str, title: str):
    if shop_id not in CATALOGS:
        raise HTTPException(status_code=404, detail="Shop not found")

    book = CATALOGS[shop_id].get(title.strip().lower())
    if not book:
        raise HTTPException(status_code=404, detail="Book not found in this shop")

    return SearchResponse(
        title=book["Title"],
        price=book["price"],
        stock=book["stock"],
        shop_id=shop_id,
        category=book["categories"],
    )


@app.post("/shops/{shop_id}/buy", response_model=BuyResponse)
async def buy_book(shop_id: str, payload: BuyRequest):
    if shop_id not in CATALOGS:
        raise HTTPException(status_code=404, detail="Shop not found")

    book = CATALOGS[shop_id].get(payload.title.strip().lower())
    if not book:
        raise HTTPException(status_code=404, detail="Book title not found in this shop")

    if not book["stock"]:
        raise HTTPException(status_code=400, detail="Book is out of stock")

    transaction_id = f"TXN-{uuid.uuid4().hex[:8].upper()}"

    return BuyResponse(
        transaction_id=transaction_id,
        status="confirmed",
        eta="3-5 business days",
    )


@app.get("/")
async def root():
    return {
        "message": "Mock Retailer API is running.",
        "shops": list(CATALOGS.keys()),
    }
