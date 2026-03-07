# BookBuying Agent Project

This project implements a simple AI agent that can recommend and purchase books.

The agent receives a natural language request from the user, finds relevant books using semantic search and an LLM, and can search partner shops to purchase the selected book.

## Main Services

### `app.py`

This is the **main application that runs on Render**.
It starts the FastAPI service used by the frontend and connects requests to the agent.

### `agent_server.py`

This file exposes the **API of the agent**.
The frontend communicates with this API in order to interact with the agent.

The agent can:

* recommend books
* search shops for prices
* buy books

### `mock_retailer/`

This folder contains a **mock implementation of book stores**.

It simulates external retailer services and exposes an API that allows the agent to:

* search for a book in different shops
* get price and stock information
* simulate purchasing a book

These endpoints are used by the agent tools.

## Other Files

* `recommendation_tool.py` – logic for recommending books using RAG and an LLM
* `find_and_buy_tools.py` – tools for searching shops and purchasing books
* `bookbuy_agent.py` – agent setup and orchestration
* `config.py` – configuration and environment variables
* `ingest.py` – script used to ingest books into the vector database (pinecone)
