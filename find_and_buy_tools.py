import os
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed
from langchain_core.tools import tool

RETAILER_API_URL = os.getenv("RETAILER_API_URL", "http://127.0.0.1:10000")
SHOPS = ["fiction_boutique", "knowledge_store", "mega_market1", "mega_market2"]


def _search_shop(shop: str, book_title: str) -> dict:
    try:
        url = f"{RETAILER_API_URL}/shops/{shop}/search"
        res = requests.get(
            url,
            params={"title": book_title},
            timeout=40,
        )

        if res.status_code != 200:
            return {
                "ok": False,
                "shop": shop,
                "error": f"HTTP {res.status_code}: {res.text}",
            }

        data = res.json()
        in_stock = bool(data.get("stock"))
        price = data.get("price")

        return {
            "ok": True,
            "offer": {
                "shop": shop,
                "price": float(price) if (price is not None and in_stock) else None,
                "in_stock": in_stock,
                "store_title": data.get("title"),
            },
        }

    except Exception as e:
        return {
            "ok": False,
            "shop": shop,
            "error": str(e),
        }


@tool("findPricesTool")
def find_prices(book_title: str) -> dict:
    """
    Search the book in all partner shops and return all offers.
    """
    offers = []
    errors = []

    with ThreadPoolExecutor(max_workers=min(4, len(SHOPS))) as executor:
        futures = [executor.submit(_search_shop, shop, book_title) for shop in SHOPS]

        for future in as_completed(futures):
            result = future.result()
            if result["ok"]:
                offers.append(result["offer"])
            else:
                errors.append({
                    "shop": result["shop"],
                    "error": result["error"],
                })

    if not any(o["in_stock"] and o["price"] is not None for o in offers):
        return {
            "status": "out_of_stock",
            "title": book_title,
            "offers": offers,
            "errors": errors,
        }

    return {
        "status": "found",
        "title": book_title,
        "offers": offers,
        "errors": errors,
    }


@tool("buyBookTool")
def buy_book(shop_id: str, book_title: str, address: str, payment_token: str) -> dict:
    """
    Buy a book from a specific shop.
    """
    try:
        buy_url = f"{RETAILER_API_URL}/shops/{shop_id}/buy"
        payload = {
            "title": book_title,
            "user_address": address,
            "payment_token": payment_token,
        }

        res = requests.post(buy_url, json=payload, timeout=40)

        if res.status_code == 200:
            data = res.json()
            return {
                "status": "success",
                "shop": shop_id,
                "title": book_title,
                **data,
            }

        return {
            "status": "failed",
            "shop": shop_id,
            "title": book_title,
            "error": f"HTTP {res.status_code}: {res.text}",
        }

    except Exception as e:
        return {
            "status": "failed",
            "shop": shop_id,
            "title": book_title,
            "error": str(e),
        }

