import json
from typing import List, Dict, Any, Optional
from pinecone import Pinecone
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_pinecone import PineconeVectorStore
from config import (OPENAI_API_KEY, OPENAI_BASE_URL, PINECONE_API_KEY, PINECONE_INDEX_NAME, LLM_MODEL, EMBEDDING_MODEL,
                    TOP_K_RETURN_BOOKS, TOP_K_REVIEWS, supabase_client)
from typing import Dict, Any
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

    return [
        {
            "title": d.metadata.get("title", ""),
            "authors": d.metadata.get("authors", []),
            "publishedDate": d.metadata.get("publishedDate", ""),
            "categories": d.metadata.get("categories", []),
            "bookLength": d.metadata.get("bookLength"),
            "description": d.page_content.split("Description:", 1)[-1],
        }
        for d in docs
    ]


def llm_select_books_by_description(user_prompt: str, rag_books: List[Dict[str, Any]],
                                    user_preferences: Optional[List[str]] = None, ) -> List[Dict[str, Any]]:
    """
    LLM step: select up to 4 books from the RAG results.
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

    You must choose books ONLY from the candidates.
    You must follow the user's preferences as strict constraints when possible.

    Rules:
    - Preferences are requirements, not suggestions.
    - If a preference conflicts with all candidates, relax ONLY the least important ones, but keep book length if specified.
    - Always filter out books that clearly violate: book length, category constraints, language constraints, etc. (if given).
    - If the user specifies a book length, prefer books in that range even if the description match is slightly weaker.

    Return 3 to 4 book titles if possible.
    Return fewer only if fewer reasonably match the request AND the preferences.

    Return ONLY valid JSON.
    Output:
    {{ "titles": ["Book Title 1", "Book Title 2"] }}
    If none match: {{ "titles": [] }}

    User request:
    {user_prompt}

    User preferences (treat as constraints):
    {user_preferences}

    Candidate books:
    {json.dumps(rag_books, ensure_ascii=False)}
    """.strip()

    response = llm.invoke(prompt)
    raw = (response.content or "").strip()
    if not raw:
        return []

    if raw.startswith("```"):
        raw = raw.strip("`").strip()
        if raw.lower().startswith("json"):
            raw = raw[4:].strip()

    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return []
    titles = set(data.get("titles", []))
    selected_books = [b for b in rag_books if b.get("title") in titles]
    print(json.dumps(selected_books, ensure_ascii=False))
    return selected_books


def attach_reviews(books):
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


def llm_choose_book_by_reviews(user_prompt: str, description_books: List[dict], ) -> str:
    """
    Chooses ONE book title based only on reviews.
    Returns the selected title.
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
    For each book, consider its description plus its reviews. Reviews matter the most for the final decision.
    Each review contains "summary_review" (review summary), "review_score" (rating for that review), and there is also a "book_overall_score" (average rating for the book) — use all of them, but prioritize the actual review content.
    
    Give strong preference to books whose reviews suggest the book is engaging and worth reading for this user,
    and avoid books whose reviews suggest the book is boring, slow, dull, or disappointing.
    If two books seem equally good by reviews, use the original book order as a tie-breaker.
    
    Return ONLY valid JSON, with no extra text.
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
    if not raw:
        return ""

    if raw.startswith("```"):
        raw = raw.strip("`").strip()
        if raw.lower().startswith("json"):
            raw = raw[4:].strip()

    try:
        result = json.loads(raw)
    except json.JSONDecodeError:
        return ""

    return result.get("title", "")


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

    Args:
        user_prompt: The user's request describing the type of book they want.
        excluded_titles: List of book titles that must NOT be recommended
                         (e.g., books already read or disliked by the user).
        user_preferences: Optional list of user preferences such as
                          "book length: between 100 and 400 pages".

     Returns:
        A dictionary with the recommendation result.

        If a suitable book is found:
        {
            "status": "found",
            "title": "Book Title",
            "author": "Author Name",
            "published_date": "YYYY",
            "categories": ["Category1", "Category2"],
            "book_length": 320,
            "description": "Short description of the book"
        }

        If no suitable book is found:
        {
            "status": "no_match"
        }
    """
    rag_books = rag_books_by_description(
        user_prompt=user_prompt,
        excluded_titles=excluded_titles,
    )

    description_books = llm_select_books_by_description(
        user_prompt=user_prompt,
        rag_books=rag_books,
        user_preferences=user_preferences,
    )
    if not description_books:
        return {"status": "no_match"}

    rating_book = llm_choose_book_by_reviews(
        user_prompt=user_prompt,
        description_books=description_books,
    )
    if not rating_book:
        return {"status": "no_match"}

    selected_book = None
    for book in description_books:
        if book.get("title") == rating_book:
            selected_book = book
            break

    if not selected_book:
        return {"status": "no_match"}

    return {
        "status": "found",
        "title": selected_book.get("title"),
        "author": selected_book.get("author"),
        "published_date": selected_book.get("publishedDate"),
        "categories": selected_book.get("categories"),
        "book_length": selected_book.get("bookLength"),
        "description": selected_book.get("description"),
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

