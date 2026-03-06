import json
from typing import List, Dict, Any, Optional, Tuple
from pinecone import Pinecone
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_pinecone import PineconeVectorStore
from config import (
    OPENAI_API_KEY,
    OPENAI_BASE_URL,
    PINECONE_API_KEY,
    PINECONE_INDEX_NAME,
    LLM_MODEL,
    EMBEDDING_MODEL,
    TOP_K_RETURN_BOOKS,
    TOP_K_REVIEWS,
    supabase_client,
)
from langchain_core.tools import tool


def get_vector_store() -> PineconeVectorStore:
    pc = Pinecone(api_key=PINECONE_API_KEY)
    index = pc.Index(PINECONE_INDEX_NAME)

    embeddings = OpenAIEmbeddings(
        model=EMBEDDING_MODEL,
        api_key=OPENAI_API_KEY,
        base_url=OPENAI_BASE_URL,
    )

    return PineconeVectorStore(
        index=index,
        embedding=embeddings,
    )


def rag_books_by_description(user_prompt: str, excluded_titles: List[str]) -> List[dict]:
    """
    RAG step:
    - Semantic search on description
    - Metadata filter on title
    - Returns exactly TOP_K_RETURN_BOOKS results (if available)
    """
    store = get_vector_store()

    pinecone_filter = (
        {"title": {"$nin": excluded_titles}}
        if excluded_titles
        else None
    )

    docs = store.similarity_search(
        query=user_prompt,
        k=TOP_K_RETURN_BOOKS,
        filter=pinecone_filter,
    )

    results = []
    for d in docs:
        page_content = d.page_content or ""
        description = (
            page_content.split("Description:", 1)[-1].strip()
            if "Description:" in page_content
            else page_content.strip()
        )

        results.append({
            "title": d.metadata.get("title", ""),
            "authors": d.metadata.get("authors", []),
            "publishedDate": d.metadata.get("publishedDate", ""),
            "categories": d.metadata.get("categories", []),
            "bookLength": d.metadata.get("bookLength"),
            "description": description,
        })

    return results


def llm_select_books_by_description(
    user_prompt: str,
    rag_books: List[Dict[str, Any]],
    user_preferences: Optional[List[str]] = None,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    LLM step: select up to 4 books from the RAG results.
    Returns:
      - selected books
      - llm steps for tracing
    """
    user_preferences = user_preferences or []

    llm = ChatOpenAI(
        model=LLM_MODEL,
        api_key=OPENAI_API_KEY,
        base_url=OPENAI_BASE_URL,
        max_tokens=1024,
        temperature=1,
    )

    prompt = f"""
    You are an expert book curator.

    Choose books ONLY from the candidate books.

    Rules:
    - Prioritize matching the user's request.
    - Consider user preferences important, but do not be overly strict.
    - If a book generally fits the request, it can still be recommended even if one preference is slightly off.
    - If a preference conflicts with all candidates, relax the least important ones (but still prefer books that follow them).
    - If book length is specified, prefer books in that range.

    Return 3–4 book titles whenever possible.
    Return fewer only if there are truly not enough reasonable matches.

    Return ONLY valid JSON.
    Output:
    {{ "titles": ["Book Title 1", "Book Title 2", "Book Title 3"] }}
    If none match: {{ "titles": [] }}

    User request:
    {user_prompt}

    User preferences:
    {user_preferences}

    Candidate books:
    {json.dumps(rag_books, ensure_ascii=False)}
    """.strip()

    response = llm.invoke(prompt)
    raw = (response.content or "").strip()

    llm_step = {
        "module": "DescriptionSelector",
        "prompt": {
            "user_prompt": user_prompt,
            "user_preferences": user_preferences,
            "candidate_books": rag_books
        },
        "response": {
            "raw_output": response.content or ""
        }
    }

    if not raw:
        return [], [llm_step]

    if raw.startswith("```"):
        raw = raw.strip("`").strip()
        if raw.lower().startswith("json"):
            raw = raw[4:].strip()

    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return [], [llm_step]

    titles = set(data.get("titles", []))
    selected_books = [b for b in rag_books if b.get("title") in titles]

    return selected_books, [llm_step]


def attach_reviews(books: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    # Handle missing Supabase client
    if not supabase_client:
        for book in books:
            book["summary_reviews"] = []
            book["avg_score"] = None
        return books

    titles = [b["title"] for b in books]

    rows = (
        supabase_client.table("books_ratings")
        .select("title,review_summary,review_score")
        .in_("title", titles)
        .execute()
        .data
    ) or []

    for book in books:
        title = book["title"]

        relevant = [r for r in rows if r["title"] == title]

        book["summary_reviews"] = [
            f"{r['review_summary']}, {r['review_score']}"
            for r in relevant
        ]

        scores = [r["review_score"] for r in relevant if r["review_score"] is not None]
        book["avg_score"] = sum(scores) / len(scores) if scores else None

    return books


def llm_choose_book_by_reviews(
    user_prompt: str,
    description_books: List[dict],
) -> Tuple[str, List[Dict[str, Any]]]:
    """
    Chooses ONE book title based on request + reviews.
    Returns:
      - selected title
      - llm steps for tracing
    """
    description_books = attach_reviews(description_books)

    llm = ChatOpenAI(
        model=LLM_MODEL,
        api_key=OPENAI_API_KEY,
        base_url=OPENAI_BASE_URL,
        max_tokens=1024,
        temperature=1,
    )

    prompt = f"""
    You are an expert reader and book curator. Choose ONE final book for the user.

    You will receive a user request and a list of candidate books (in ranked order from the previous step).

    For each book:
    - First ensure it fits the user's request.
    - Then use the reviews to help choose the best option.

    Reviews are important for the final decision.
    Each review includes "summary_review" and "review_score", and each book may also have a "book_overall_score".
    Prefer books whose reviews suggest the book is engaging and worthwhile.

    However:
    - A book should NOT be rejected only because it has few or no reviews.
    - If two books fit the request similarly, prefer the one with stronger reviews.

    If reviews are similar or missing, use the original candidate order as a tie-breaker.

    Return ONLY valid JSON.
    Output format:
    {{
      "title": "Book Title"
    }}

    If none of the candidate books reasonably match, return:
    {{ "title": "" }}

    User request:
    {user_prompt}

    Candidate books with reviews:
    {json.dumps(description_books, ensure_ascii=False)}
    """.strip()

    response = llm.invoke(prompt)
    raw = (response.content or "").strip()

    llm_step = {
        "module": "ReviewFinalSelector",
        "prompt": {
            "user_prompt": user_prompt,
            "candidate_books_with_reviews": description_books
        },
        "response": {
            "raw_output": response.content or ""
        }
    }

    if not raw:
        return "", [llm_step]

    if raw.startswith("```"):
        raw = raw.strip("`").strip()
        if raw.lower().startswith("json"):
            raw = raw[4:].strip()

    try:
        result = json.loads(raw)
    except json.JSONDecodeError:
        return "", [llm_step]

    return result.get("title", ""), [llm_step]


@tool("recommendationTool")
def recommendation_tool(
    user_prompt: str,
    excluded_titles: List[str],
    user_preferences: Optional[List[str]] = None
) -> dict:
    """
    Recommend a single book title for the user.

    This tool searches for books using semantic search (RAG) and then uses an LLM
    to select the best matching book based on the user's request, preferences,
    and reviews.

    Returns:
        If found:
        {
            "status": "found",
            "title": "Book Title",
            "authors": ["Author Name"],
            "published_date": "YYYY",
            "categories": ["Category1", "Category2"],
            "book_length": 320,
            "description": "Short description of the book",
            "llm_steps": [...]
        }

        If not found:
        {
            "status": "no_match",
            "llm_steps": [...]
        }
    """
    llm_steps: List[Dict[str, Any]] = []

    rag_books = rag_books_by_description(
        user_prompt=user_prompt,
        excluded_titles=excluded_titles,
    )

    description_books, description_steps = llm_select_books_by_description(
        user_prompt=user_prompt,
        rag_books=rag_books,
        user_preferences=user_preferences,
    )
    llm_steps.extend(description_steps)

    if not description_books:
        return {
            "status": "no_match",
            "llm_steps": llm_steps,
        }

    rating_book, review_steps = llm_choose_book_by_reviews(
        user_prompt=user_prompt,
        description_books=description_books,
    )
    llm_steps.extend(review_steps)

    if not rating_book:
        return {
            "status": "no_match",
            "llm_steps": llm_steps,
        }

    selected_book = None
    for book in description_books:
        if book.get("title") == rating_book:
            selected_book = book
            break

    if not selected_book:
        return {
            "status": "no_match",
            "llm_steps": llm_steps,
        }

    return {
        "status": "found",
        "title": selected_book.get("title"),
        "authors": selected_book.get("authors"),
        "published_date": selected_book.get("publishedDate"),
        "categories": selected_book.get("categories"),
        "book_length": selected_book.get("bookLength"),
        "description": selected_book.get("description"),
        "llm_steps": llm_steps,
    }


if __name__ == "__main__":
    user_prompt = "Im looking for a scholarly Christian book on the biblical theology and purpose of the Church."
    disliked_titles = []
    already_read_titles = ["Wonderful Worship in Smaller Churches"]
    user_preferences = ["book length: between 100 and 400"]
    excluded_titles = list(set(disliked_titles + already_read_titles))

    book = recommendation_tool(
        user_prompt=user_prompt,
        excluded_titles=excluded_titles,
        user_preferences=user_preferences,
    )

    print(json.dumps(book, ensure_ascii=False, indent=2))

