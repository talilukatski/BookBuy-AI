from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import List, Any, Optional, Dict
import os
from bookbuy_agent import UserPersonalDetails, BookBuyAgentRunner
from config import OPENAI_API_KEY, OPENAI_BASE_URL, LLM_MODEL
from langchain_openai import ChatOpenAI


app = FastAPI()

# Configure CORS - Allow ALL origins for Render deployment
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# --- Data Models (Schemas) ---
class Student(BaseModel):
    name: str
    email: str


class TeamInfo(BaseModel):
    group_batch_order_number: str
    team_name: str
    students: List[Student]


class AgentInfo(BaseModel):
    description: str
    purpose: str
    prompt_template: List[Dict[str, Any]]
    prompt_examples: List[Dict[str, Any]]


class ExecuteRequest(BaseModel):
    prompt: str
    address: str
    payment_token: str
    book_preferences: Optional[List[str]] = None
    disliked_titles: Optional[List[str]] = None
    already_read_titles: Optional[List[str]] = None


class ExecuteResponse(BaseModel):
    status: str
    error: Optional[str]
    response: Optional[str] 
    steps: List[Dict[str, Any]]


# --- API Endpoints ---
@app.get("/api/team_info", response_model=TeamInfo)
async def get_team_info():
    """Returns the team details."""
    return {
        "group_batch_order_number": "1_3",
        "team_name": "BookBuy AI",
        "students": [
            {"name": "Daniel Shmulevich", "email": "dshmulevich@campus.technion.ac.il"},
            {"name": "Antal Lykazky", "email": "antallykazky@campus.technion.ac.il"},
            {"name": "Offek Hai", "email": "offek.hai@campus.technion.ac.il"}
        ]
    }


@app.get("/api/agent_info", response_model=AgentInfo)
async def get_agent_info():
    """Returns agent metadata."""
    return {
        "description": "BookBuyAI is an autonomous AI agent that independently selects a suitable book, finds the lowest available price across shops, and completes the purchase automatically.",
        "purpose": "To provide a fully automated book-buying process: BookBuyAI chooses a book on its own, compares prices across stores, buys the cheapest option, and may decide to skip a book and search for another if the price seems too expensive.",
        "prompt_template": [
            {
                "prompt": "I'm looking for a theoretical astrology book that explains astrological patterns",
                "address": "Nofit Hol, Haifa",
                "payment_token": "1234567",
                "book_preferences": [
                    "astrology", "theoretical approach", "meditation exercises", "visual explanations",
                    "book length: around 400 pages"
                ],
                "disliked_titles": [
                    "The Forbidden Stories of Marta Veneranda"
                ],
                "already_read_titles": [
                    "The House on Mango Street"
                ]
            },
            {
                "prompt": "I'm looking for an interesting book about wildlife that explains how animals survive and interact in nature",
                "address": "Tel Aviv",
                "payment_token": "9876543",
                "book_preferences": [
                    "animals", "wildlife", "nature", "animal behavior",
                    "book length: around 150-300 pages"
                ],
                "disliked_titles": [],
                "already_read_titles": [
                    "The Jungle Book",
                    "Life of Pi"
                ]
            },
            {
                "prompt": "I'm looking for an interesting book about wildlife that explains how animals survive and interact in nature",
                "address": "Tel Aviv",
                "payment_token": "9876543",
                "book_preferences": [
                    "book length: around 30 pages"
                ],
                "disliked_titles": [],
                "already_read_titles": [
                    "The Jungle Book",
                    "Life of Pi"
                ]
            }
        ],
        "prompt_examples": [
            {
                "prompt": {
                    "prompt": "I'm looking for a theoretical astrology book that explains astrological patterns",
                    "address": "Nofit Hol, Haifa",
                    "payment_token": "1234567",
                    "book_preferences": [
                        "astrology", "theoretical approach", "meditation exercises", "visual explanations",
                        "book length: around 400 pages"
                    ],
                    "disliked_titles": [
                        "The Forbidden Stories of Marta Veneranda"
                    ],
                    "already_read_titles": [
                        "The House on Mango Street"
                    ]
                },
                "full_response": "Success! Bought 'Astrology Inside Out' from mega_market (Txn: TXN-53290F2D). Estimated delivery: 3-5 business days to Nofit Hol, Haifa.",
                "steps": [
                    {
                        "attempt": 1,
                        "module": "recommendationTool",
                        "prompt": {
                            "user_prompt": "I want a theoretical astrology book that explains astrological patterns. Prefer a theoretical approach with clear explanations of patterns, visual explanations/diagrams, and possibly meditation exercises. Book length around 400 pages.",
                            "excluded_titles": [
                                "The House on Mango Street",
                                "The Forbidden Stories of Marta Veneranda"
                            ],
                            "user_preferences": [
                                "astrology",
                                "theoretical approach",
                                "meditation exercises",
                                "visual explanations",
                                "book length: around 400 pages"
                            ]
                        },
                        "response": {
                            "status": "found",
                            "title": "Astrology Inside Out",
                            "author": None,
                            "published_date": "1982-03",
                            "categories": "Astrology",
                            "book_length": 361,
                            "description": " This excellent introduction to astrology offers a real departure from most books on the subject. Using a theoretical approach, author Bruce Nevin combines esoteric traditions with modern harmonic research and recent developments in physics and psychology. Meditation exercises and ingenious visuals teach beginning, as well as seasoned astrologers, new ways to recognize and interpret astrological patterns from \"inside out.\" Use the meditation exercises for self-examination and psychic development. You can also build a self-reliant, intuitive approach to chart interpretation."
                        }
                    },
                    {
                        "attempt": 1,
                        "module": "findPricesTool",
                        "prompt": {
                            "book_title": "Astrology Inside Out"
                        },
                        "response": {
                            "status": "found",
                            "title": "Astrology Inside Out",
                            "offers": [
                                {
                                    "shop": "mega_market",
                                    "price": 73.72,
                                    "in_stock": True,
                                    "store_title": "Astrology Inside Out"
                                }
                            ],
                            "errors": [
                                {
                                    "shop": "fiction_boutique",
                                    "error": "HTTP 404: {\"detail\":\"Book not found in this shop\"}"
                                },
                                {
                                    "shop": "knowledge_store",
                                    "error": "HTTP 404: {\"detail\":\"Book not found in this shop\"}"
                                }
                            ]
                        }
                    },
                    {
                        "attempt": 1,
                        "module": "buyBookTool",
                        "prompt": {
                            "shop_id": "mega_market",
                            "book_title": "Astrology Inside Out",
                            "address": "Nofit Hol, Haifa",
                            "payment_token": "1234567"
                        },
                        "response": {
                            "status": "confirmed",
                            "shop": "mega_market",
                            "title": "Astrology Inside Out",
                            "transaction_id": "TXN-53290F2D",
                            "eta": "3-5 business days"
                        }
                    }
                ]
            },
            {
                "prompt": {
                    "prompt": "I'm looking for an interesting book about wildlife that explains how animals survive and interact in nature",
                    "address": "Tel Aviv",
                    "payment_token": "9876543",
                    "book_preferences": [
                        "animals", "wildlife", "nature", "animal behavior",
                        "book length: around 150-300 pages"
                    ],
                    "disliked_titles": [],
                    "already_read_titles": [
                        "The Jungle Book",
                        "Life of Pi"
                    ]
                },
                "full_response": "Success! Bought 'Prey' from mega_market (Txn: TXN-D9218151). Estimated delivery: 3-5 business days to Tel Aviv.",
                "steps": [
                    {
                        "attempt": 1,
                        "module": "recommendationTool",
                        "prompt": {
                            "user_prompt": "I'm looking for an interesting book about wildlife that explains how animals survive and interact in nature",
                            "excluded_titles": [
                                "Life of Pi",
                                "The Jungle Book"
                            ],
                            "user_preferences": [
                                "animals",
                                "wildlife",
                                "nature",
                                "animal behavior",
                                "book length: around 150-300 pages"
                            ]
                        },
                        "response": {
                            "status": "found",
                            "title": "Prey",
                            "author": None,
                            "published_date": "2000",
                            "categories": "Predation (Biology)",
                            "book_length": 246,
                            "description": " Tropical rainforests cover only about 6% of the earth's surface, yet they are home to more than half the world's species. This book looks at the biology and behaviour of predators and prey describing how they live together in harmony and balance and how vital they are to the wellbeing of the whole world. It is one of a series of books which aims to provide an understanding of the unique ecosystems of the rainforest. It describes the amazing plants and animals and how they interact, and addresses local and global environmental and conservation issues."
                        }
                    },
                    {
                        "attempt": 1,
                        "module": "findPricesTool",
                        "prompt": {
                            "book_title": "Prey"
                        },
                        "response": {
                            "status": "found",
                            "title": "Prey",
                            "offers": [
                                {
                                    "shop": "knowledge_store",
                                    "price": None,
                                    "in_stock": False,
                                    "store_title": "Modelling Waffen-SS Figures (Osprey Modelling)"
                                },
                                {
                                    "shop": "mega_market",
                                    "price": 44.5,
                                    "in_stock": True,
                                    "store_title": "Prey"
                                }
                            ],
                            "errors": [
                                {
                                    "shop": "fiction_boutique",
                                    "error": "HTTP 404: {\"detail\":\"Book not found in this shop\"}"
                                }
                            ]
                        }
                    },
                    {
                        "attempt": 1,
                        "module": "buyBookTool",
                        "prompt": {
                            "shop_id": "mega_market",
                            "book_title": "Prey",
                            "address": "Tel Aviv",
                            "payment_token": "9876543"
                        },
                        "response": {
                            "status": "confirmed",
                            "shop": "mega_market",
                            "title": "Prey",
                            "transaction_id": "TXN-D9218151",
                            "eta": "3-5 business days"
                        }
                    }
                ]
            },
            {
                "prompt": {
                    "prompt": "I'm looking for an interesting book about wildlife that explains how animals survive and interact in nature",
                    "address": "Tel Aviv",
                    "payment_token": "9876543",
                    "book_preferences": [
                        "book length: around 30 pages"
                    ],
                    "disliked_titles": [],
                    "already_read_titles": [
                        "The Jungle Book",
                        "Life of Pi"
                    ]
                },
                "response": "Sorry — I couldn't find any suitable recommendation for your request, so I didn’t proceed to purchase attempts. Try broadening the topic or updating your preferences, and I’ll try again.",
                "steps": [
                    {
                        "attempt": 1,
                        "module": "recommendationTool",
                        "prompt": {
                            "user_prompt": "I'm looking for an interesting book about wildlife that explains how animals survive and interact in nature",
                            "excluded_titles": [
                                "Life of Pi",
                                "The Jungle Book"
                            ],
                            "user_preferences": [
                                "book length: around 30 pages"
                            ]
                        },
                        "response": {
                            "status": "no_match"
                        }
                    }
                ]
            }


        ]
    }


@app.get("/api/model_architecture")
async def get_model_architecture():
    """Returns the architecture diagram."""
    file_path = "architecture.png"
    if os.path.exists(file_path):
        return FileResponse(file_path, media_type="image/png")
    else:
        raise HTTPException(status_code=404, detail="Architecture diagram not found.")


@app.post("/api/execute", response_model=ExecuteResponse)
async def execute_agent(request: ExecuteRequest):
    steps = []
    try:
        user = UserPersonalDetails(
            user_preferences=request.book_preferences,
            disliked_titles=request.disliked_titles,
            already_read_titles=request.already_read_titles,
            address=request.address,
            payment_token=request.payment_token
        )

        llm = ChatOpenAI(
            model=LLM_MODEL,
            api_key=OPENAI_API_KEY,
            base_url=OPENAI_BASE_URL,
            max_tokens=1024,
        )
        import find_and_buy_tools
        find_and_buy_tools.CURRENT_USER = user

        runner = BookBuyAgentRunner(llm, user)
        result = runner.run(request.prompt)

        return {
            "status": result["status"],
            "error": None,
            "response": result["response"],
            "steps": result["steps"],
        }

    except Exception as e:
        return {
            "status": "error",
            "error": str(e),
            "response": None,
            "steps": steps,
        }

if __name__ == "__main__":
    import uvicorn
    # Agent runs on 8080. Store runs on 8000.

    uvicorn.run(app, host="0.0.0.0", port=8080)
