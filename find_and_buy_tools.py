import os
import requests
from langchain_core.tools import tool


# --- Configuration for Retailer API ---
# This assumes the Mock Retailer API is running on port 8000
RETAILER_API_URL = os.getenv("RETAILER_API_URL", "http://127.0.0.1:8000")
SHOPS = ["fiction_boutique", "knowledge_store", "mega_market1", "mega_market2"]

@tool("findPricesTool")
def find_prices(book_title: str) -> dict:
    """
    Search the book in all partner shops and return ALL offers.
    The agent will choose which shop to buy from.

    Args:
        book_title: The title to search for.

    Returns:
        {
          "status": "offers" | "out_of_stock" | "error",
          "title": "<requested title>",
          "offers": [
             {"shop": "...", "price": 12.3, "in_stock": true, "store_title": "..."},
             ...
          ],
          "errors": [{"shop":"...", "error":"..."}]   # optional
        }
    """
    offers = []
    errors = []

    for shop in SHOPS:
        try:
            url = f"{RETAILER_API_URL}/shops/{shop}/search"
            res = requests.get(url, params={"title": book_title}, timeout=10)

            if res.status_code != 200:
                errors.append({"shop": shop, "error": f"HTTP {res.status_code}: {res.text}"})
                continue

            data = res.json()

            # Normalize fields from retailer response
            in_stock = bool(data.get("stock"))
            price = data.get("price", None)

            offers.append({
                "shop": shop,
                "price": float(price) if (price is not None and in_stock) else None,
                "in_stock": in_stock,
                "store_title": data.get("title")  # exact title as returned by the shop
            })

        except Exception as e:
            errors.append({"shop": shop, "error": str(e)})

    # If no in-stock offers exist, return out_of_stock
    if not any(o["in_stock"] and o["price"] is not None for o in offers):
        return {
            "status": "out_of_stock",
            "title": book_title,
            "offers": offers,
            "errors": errors
        }

    return {
        "status": "found",
        "title": book_title,
        "offers": offers,
        "errors": errors
    }


@tool("buyBookTool")
def buy_book(shop_id: str, book_title: str, address: str, payment_token: str) -> dict:
    """
    Buy a book from a specific shop.

    Args:
        shop_id: Shop identifier.
        book_title: The title to buy (should match the shop's expected title if possible).
        address: The user's shipping address.
        payment_token: Token representing the user's payment method.

    Returns:
        Success:
          {"status":"success","transaction_id":"...","shop":"...","title":"..."}
        Failure:
          {"status":"failed","shop":"...","title":"...","error":"..."}
    """
    try:
        buy_url = f"{RETAILER_API_URL}/shops/{shop_id}/buy"
        payload = {
            "title": book_title,
            "user_address": address,
            "payment_token": payment_token
        }

        res = requests.post(buy_url, json=payload, timeout=15)

        if res.status_code == 200:
            data = res.json()
            # Keep original response but add a stable status for the agent
            return {
                "status": "success",
                "shop": shop_id,
                "title": book_title,
                **data
            }

        return {
            "status": "failed",
            "shop": shop_id,
            "title": book_title,
            "error": f"HTTP {res.status_code}: {res.text}"
        }

    except Exception as e:
        return {
            "status": "failed",
            "shop": shop_id,
            "title": book_title,
            "error": str(e)

        }

