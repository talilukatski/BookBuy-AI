from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uuid
import pandas as pd
import os
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = FastAPI(title="Mock Retailer API")

# --- Data Loading ---

CATALOG_DIR = os.path.join(os.path.dirname(__file__), "catalogs")
CATALOGS = {}

def load_catalogs():
    """Loads CSV catalogs into global dictionary."""
    global CATALOGS
    shops = ["fiction_boutique", "knowledge_store", "mega_market1", "mega_market2"]
    
    print("Loading catalogs...")
    for shop in shops:
        file_path = os.path.join(CATALOG_DIR, f"{shop}.csv")
        if os.path.exists(file_path):
            try:
                df = pd.read_csv(file_path)
                # Ensure efficient string matching
                df['title_lower'] = df['Title'].astype(str).str.lower()
                CATALOGS[shop] = df
                logger.info(f"  - Loaded {shop} ({len(df)} items)")
            except Exception as e:
                logger.error(f"  - Error loading {shop}: {e}")
        else:
            logger.warning(f"  - Warning: {file_path} not found.")

# Load on startup (or module import for simplicity in this mock)
load_catalogs()

# --- Data Models ---

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

# --- Endpoints ---

@app.get("/shops/{shop_id}/search", response_model=SearchResponse)
async def search_book(shop_id: str, title: str):
    """
    Search for a book by title in a specific shop.
    Returns the first approximate match.
    """
    if shop_id not in CATALOGS:
        raise HTTPException(status_code=404, detail="Shop not found")
    
    df = CATALOGS[shop_id]
    
    # Case-insensitive partial match
    # We look for the search term in the 'title_lower' column
    match = df[df['title_lower'].str.contains(title.lower(), na=False, regex=False)]
    
    if match.empty:
        raise HTTPException(status_code=404, detail="Book not found in this shop")
    
    # Return the first match
    book = match.iloc[0]
    
    return SearchResponse(
        title=book['Title'],
        price=float(book['price']),
        stock=bool(book['stock']),
        shop_id=shop_id,
        category=str(book['categories']) if 'categories' in book else "Unknown"
    )

@app.post("/shops/{shop_id}/buy", response_model=BuyResponse)
async def buy_book(shop_id: str, payload: BuyRequest):
    """
    Simulate purchasing a book.
    """
    if shop_id not in CATALOGS:
        raise HTTPException(status_code=404, detail="Shop not found")

    df = CATALOGS[shop_id]
    
    # Verify book exists exactly (or close enough for this mock)
    # We'll use exact match on Title for buying to be safe, or stick to the search logic.
    # Let's align with search: find the book by exact title ignoring case
    match = df[df['title_lower'] == payload.title.lower()]
    
    if match.empty:
         raise HTTPException(status_code=404, detail="Book title not found in this shop")
    
    book = match.iloc[0]
    
    if not book["stock"]:
        raise HTTPException(status_code=400, detail="Book is out of stock")

    # Simulate processing
    transaction_id = f"TXN-{uuid.uuid4().hex[:8].upper()}"
    
    return BuyResponse(
        transaction_id=transaction_id,
        status="confirmed",
        eta="3-5 business days"
    )

@app.get("/")
async def root():
    return {
        "message": "Mock Retailer API is running.",
        "shops": list(CATALOGS.keys())
    }
