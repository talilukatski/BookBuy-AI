import pandas as pd
from tqdm import tqdm
from dotenv import load_dotenv
import pinecone
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_text_splitters import TokenTextSplitter
from langchain_core.documents import Document
from typing import List

from config import (
    OPENAI_API_KEY,
    OPENAI_BASE_URL,
    PINECONE_API_KEY,
    PINECONE_INDEX_NAME,
    EMBEDDING_MODEL,
    CHUNK_SIZE,
    OVERLAP_RATIO,
)

load_dotenv()


def load_dataframe(csv_path: str) -> pd.DataFrame:
    """Loads the TED CSV file."""
    df = pd.read_csv(csv_path)
    required_cols = ["title", "description", "authors", "publishedDate", "categories", "bookLength"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in CSV: {missing}")
    return df


def build_documents(df: pd.DataFrame) -> List[Document]:
    splitter = TokenTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=int(CHUNK_SIZE * OVERLAP_RATIO),
    )

    documents: List[Document] = []

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Building documents"):
        description = row["description"]
        if not isinstance(description, str) or not description.strip():
            continue

        title = (row["title"] or "").strip()
        print(title)

        authors = (row["authors"] or "").strip()
        categories = (row["categories"] or "").strip()

        text_for_embedding = (
            f"Title: {title}, Categories: {categories}\n Description: {description}"
        )

        base_metadata = {
            "title": title,
            "authors": authors,
            "categories": categories,
            "publishedDate": row.get("publishedDate", ""),
            "bookLength": row.get("bookLength", None),
        }

        book_docs = splitter.create_documents(
            texts=[text_for_embedding],
            metadatas=[base_metadata],
        )
        documents.extend(book_docs)

    return documents


def get_pinecone_vectorstore(embeddings: OpenAIEmbeddings) -> PineconeVectorStore:
    """ Connects to an existing Pinecone index and uses it as a LangChain vectorstore."""
    if not PINECONE_API_KEY:
        raise ValueError("PINECONE_API_KEY is not set")

    pc = pinecone.Pinecone(api_key=PINECONE_API_KEY)
    index = pc.Index(PINECONE_INDEX_NAME)

    vectorstore = PineconeVectorStore(
        index=index,
        embedding=embeddings,
    )
    return vectorstore


def main():
    if not OPENAI_API_KEY:
        raise ValueError("OPENAI_API_KEY is not set")

    print("🔹 Loading CSV...")
    df = load_dataframe("data/prepared_books_data.csv")

    print("🔹 Building Documents (splitting into chunks)...")
    documents = build_documents(df)
    print(f"Total chunks (TEST): {len(documents)}")

    print("🔹 Initializing OpenAI embeddings...")
    embeddings = OpenAIEmbeddings(
        model=EMBEDDING_MODEL,
        api_key=OPENAI_API_KEY,
        base_url=OPENAI_BASE_URL,
    )

    print("🔹 Connecting to Pinecone index...")
    vectorstore = get_pinecone_vectorstore(embeddings)

    print(f"🔹 Upserting documents into Pinecone (TEST subset)... {len(documents)}")
    vectorstore.add_documents(documents)

    print("🔹 Checking index stats...")
    index = vectorstore.index
    stats = index.describe_index_stats()
    print("Index stats:", stats)


if __name__ == "__main__":
    main()
