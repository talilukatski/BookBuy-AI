import os
from dotenv import load_dotenv
from supabase import create_client

load_dotenv()

EMBEDDING_MODEL = "RPRTHPB-text-embedding-3-small"
LLM_MODEL = "RPRTHPB-gpt-5-mini"

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL")

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME")

supabase_url = os.environ.get("SUPABASE_URL")
supabase_key = os.environ.get("SUPABASE_API_KEY")

if supabase_url and supabase_key:
    try:
        supabase_client = create_client(supabase_url, supabase_key)
    except Exception as e:
        print(f"Warning: Failed to initialize Supabase client: {e}")
        supabase_client = None
else:
    print("Warning: SUPABASE_URL or SUPABASE_API_KEY not found in environment.")
    supabase_client = None


# RAG parameters
CHUNK_SIZE = 300
OVERLAP_RATIO = 0.1
TOP_K_RETURN_BOOKS = 7
TOP_K_REVIEWS = 5
