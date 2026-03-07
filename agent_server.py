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
    user_preferences: Optional[List[str]] = None
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
                "prompt": "I'm looking for an interesting book about wildlife that explains how animals survive and interact in nature",
                "address": "Tel Aviv",
                "payment_token": "9876543",
                "user_preferences": [
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
                "prompt": "I'm looking for an interesting book about tali daniel and offek",
                "address": "Tel Aviv",
                "payment_token": "9876543",
                "user_preferences": [],
                "disliked_titles": [],
                "already_read_titles": []
            }
        ],
        "prompt_examples": [
            {
                "prompt": {
                    "prompt": "I'm looking for an interesting book about wildlife that explains how animals survive and interact in nature",
                    "address": "Tel Aviv",
                    "payment_token": "9876543",
                    "user_preferences": [
                        "animals", "wildlife", "nature", "animal behavior",
                        "book length: around 150-300 pages"
                    ],
                    "disliked_titles": [],
                    "already_read_titles": [
                        "The Jungle Book",
                        "Life of Pi"
                    ]
                },
                "full_response": "Success! Bought 'How Animals Live (The Amazing World of Animals in the Wild)' from fiction_boutique (Txn: TXN-779FF196). Estimated delivery: 3-5 business days to Tel Aviv.",
                "steps": [
                    {
                        "module": "BookBuyAgentRunner",
                        "prompt": {
                            "system": "\n    You are a ReAct BookBuy agent.\n\n    You have tools:\n    - recommendationTool(user_prompt, excluded_titles, user_preferences) -> dict\n    - findPricesTool(book_title) -> dict (returns ALL offers)\n    - buyBookTool(shop_id, book_title, address, payment_token) -> dict\n\n    You have EXACTLY 3 attempts total.\n    An attempt means: pick ONE candidate book and try to complete the whole process:\n    recommendationTool -> findPricesTool -> buyBookTool.\n\n    Attempt rules (within ONE attempt):\n    1) Call recommendationTool using the excluded_titles and user_preferences given in CONTEXT.\n    2) If recommendationTool returns status=\"no_match\": STOP ENTIRE RUN immediately.\n    3) Call findPricesTool with the recommendation.title .\n    4) If findPricesTool returns status=\"out_of_stock\" or status=\"error\": STOP this attempt immediately.\n    5) If findPricesTool returns status=\"found\":\n       - consider only offers with in_stock=true and price not None\n       - choose the lowest price\n       - If the lowest price is around or above 200 ILS, consider it expensive for a book and stop this attempt; otherwise proceed to purchase.\n       - when calling buyBookTool, prefer offer.store_title if present, else use the recommended title\n    6) Call buyBookTool exactly once.\n    7) If buyBookTool returns status=\"success\": you are DONE (final success).\n    8) If buyBookTool returns status=\"failed\": STOP this attempt immediately.\n\n    IMPORTANT:\n    - Do NOT retry within the same attempt. If something fails, stop the attempt.\n\n    CONTEXT:\n    attempt_number: 1 / 3\n    excluded_titles: [\"The Jungle Book\", \"Life of Pi\"]\n    user_preferences: [\"animals\", \"wildlife\", \"nature\", \"animal behavior\", \"book length: around 150-300 pages\"]\n    address: Tel Aviv\n    payment_token: 9876543\n    ",
                            "user": "I'm looking for an interesting book about wildlife that explains how animals survive and interact in nature"
                        },
                        "response": {
                            "content": "",
                            "tool_calls": [
                                {
                                    "name": "recommendationTool",
                                    "args": {
                                        "user_prompt": "I'm looking for an interesting book about wildlife that explains how animals survive and interact in nature",
                                        "excluded_titles": [
                                            "The Jungle Book",
                                            "Life of Pi"
                                        ],
                                        "user_preferences": [
                                            "animals",
                                            "wildlife",
                                            "nature",
                                            "animal behavior",
                                            "book length: around 150-300 pages"
                                        ]
                                    },
                                    "id": "call_0NXt9fEoFLzaaXSRPd0BBhtx",
                                    "type": "tool_call"
                                }
                            ]
                        }
                    },
                    {
                        "module": "DescriptionSelector",
                        "prompt": {
                            "user_prompt": "I'm looking for an interesting book about wildlife that explains how animals survive and interact in nature",
                            "user_preferences": [
                                "animals",
                                "wildlife",
                                "nature",
                                "animal behavior",
                                "book length: around 150-300 pages"
                            ],
                            "candidate_books": [
                                {
                                    "title": "Dinnertime for animals (Books for young explorers)",
                                    "authors": "Jane R. McCauley",
                                    "publishedDate": "1991",
                                    "categories": "Animals",
                                    "bookLength": 277,
                                    "description": "Explains how different animals living in the wild have divergent diets, eating habits, and methods of gathering food for survival."
                                },
                                {
                                    "title": "Coyotes: Predators and Survivors",
                                    "authors": "Charles L. Cadieux",
                                    "publishedDate": "1983",
                                    "categories": "Science",
                                    "bookLength": 264,
                                    "description": "A comprehensive book about the controversial coyote includes sections on its life history, present status, effect of predation on other wildlife, and continuing adaptability to mankind"
                                },
                                {
                                    "title": "How Animals Live (The Amazing World of Animals in the Wild)",
                                    "authors": "Bernard Stonehouse, Esther Bertram",
                                    "publishedDate": "2004",
                                    "categories": "Juvenile Nonfiction",
                                    "bookLength": 266,
                                    "description": "Presents an overview of the characteristic behavior, habits, and survival skills of various types of animals around the world.",
                                    "summary_reviews": [],
                                    "avg_score": None
                                },
                                {
                                    "title": "Wildlife Biology",
                                    "authors": "Raymond Fredric Dasmann, Raymond F. Dasmann",
                                    "publishedDate": "1981-01-19",
                                    "categories": "Nature",
                                    "bookLength": 339,
                                    "description": "What good are wild animals?; Wildlife values; Wildlife as a natural resource; The status of wildlife conservation; The ten-thousand year war; Everything tied together; Functioning of ecosystems; Ecosystem development; Change and degradation; Distribution of biotic communities; Ecologic niches; A place to live; Enough to eat; Cover; Water; Quantity versus interspresion; Limiting factors; Introducing wildlife populations; Density and biomass; Population structure; Natality; Mortality; Interaction of population; Characteristics; Turnover; Productivity; Territory and travels; Movements internal to the population area; Movements external to the population; Group size and spacing; Territory; Significance of spacing and movements; Too many mice, too few elephants; New populations in new habitats; Some real life problems; The annual cycle of populations; Shootable surplus; Stocking of game; Stability of populations; Evolutionary strategies; Levelling off; The many meanings of carrying capacity; Declining diversity; Species extinction and area size; Destruction of habitat; Nature reserves; Animal trade; International assistance; The controllers."
                                },
                                {
                                    "title": "Prey",
                                    "authors": "Michael Chinery",
                                    "publishedDate": "2000",
                                    "categories": "Predation (Biology)",
                                    "bookLength": 246,
                                    "description": "Tropical rainforests cover only about 6% of the earth's surface, yet they are home to more than half the world's species. This book looks at the biology and behaviour of predators and prey describing how they live together in harmony and balance and how vital they are to the wellbeing of the whole world. It is one of a series of books which aims to provide an understanding of the unique ecosystems of the rainforest. It describes the amazing plants and animals and how they interact, and addresses local and global environmental and conservation issues.",
                                    "summary_reviews": [],
                                    "avg_score": None
                                },
                                {
                                    "title": "Second Nature : Environmental Enrichment for Captive Animals (Zoo & Aquarium Biology & Conservation Ser.)",
                                    "authors": "David J. Shepherdson, Jill D. Mellen, Michael Hutchins",
                                    "publishedDate": "2012-01-11",
                                    "categories": "Nature",
                                    "bookLength": 224,
                                    "description": "Growing recognition of the complexity of animals' physical, social, and psychological lives in the wild has led both zookeepers and the zoo-going public to call for higher environmental standards for animals in captivity. Bringing together the work of animal behaviorists, zoo biologists, and psychologists, Second Nature explores a range of innovative strategies for environmental enrichment in laboratories and marine parks, as well as in zoos. From artificial fleeing-prey devices for leopards to irregular feeding schedules for whales, the practices discussed have resulted in healthier, more relaxed animals that can breed more easily and can exert some control over their environments. Moving beyond the usual studies of primates to consider the requirements of animals as diverse as reptiles, amphibians, marine mammals, small cats, hooved grazers, and bears, contributors argue that whether an animal forages in the wild or plays computer games in captivity, the satisfaction its activity provides—rather than the activity itself—determines the animal's level of physical and psychological well-being. Second Nature also discusses the ways in which environmental enrichment can help zoo-bred animals develop the stamina and adaptability for survival in the wild, and how it can produce healthier lab animals that yield more valid test results. Providing a theoretical framework for the science of environmental enrichment in a variety of settings, the book renews and extends"
                                },
                                {
                                    "title": "Animal ecology (Biology series)",
                                    "authors": "Luigi Boitani, T. K. Fuller",
                                    "publishedDate": "2000-01",
                                    "categories": "Science",
                                    "bookLength": 284,
                                    "description": "This book provides an analysis of frequently used research techniques in animal ecology, identifying their limitations and misuses, as well as possible solutions to avoid such pitfalls. The contributors provide an overarching account of central theoretical and methodological controversies. The editors have forged comprehensive presentations of key topics in animal ecology, such as territory and home range estimates, habitation evaluation, population viability analysis, GIS mapping, and measuring the dynamics of societies.",
                                    "summary_reviews": [],
                                    "avg_score": None
                                }
                            ]
                        },
                        "response": {
                            "raw_output": "{ \"titles\": [\"How Animals Live (The Amazing World of Animals in the Wild)\", \"Prey\", \"Animal ecology (Biology series)\"] }"
                        }
                    },
                    {
                        "module": "ReviewFinalSelector",
                        "prompt": {
                            "user_prompt": "I'm looking for an interesting book about wildlife that explains how animals survive and interact in nature",
                            "candidate_books_with_reviews": [
                                {
                                    "title": "How Animals Live (The Amazing World of Animals in the Wild)",
                                    "authors": "Bernard Stonehouse, Esther Bertram",
                                    "publishedDate": "2004",
                                    "categories": "Juvenile Nonfiction",
                                    "bookLength": 266,
                                    "description": "Presents an overview of the characteristic behavior, habits, and survival skills of various types of animals around the world.",
                                    "summary_reviews": [],
                                    "avg_score": None
                                },
                                {
                                    "title": "Prey",
                                    "authors": "Michael Chinery",
                                    "publishedDate": "2000",
                                    "categories": "Predation (Biology)",
                                    "bookLength": 246,
                                    "description": "Tropical rainforests cover only about 6% of the earth's surface, yet they are home to more than half the world's species. This book looks at the biology and behaviour of predators and prey describing how they live together in harmony and balance and how vital they are to the wellbeing of the whole world. It is one of a series of books which aims to provide an understanding of the unique ecosystems of the rainforest. It describes the amazing plants and animals and how they interact, and addresses local and global environmental and conservation issues.",
                                    "summary_reviews": [],
                                    "avg_score": None
                                },
                                {
                                    "title": "Animal ecology (Biology series)",
                                    "authors": "Luigi Boitani, T. K. Fuller",
                                    "publishedDate": "2000-01",
                                    "categories": "Science",
                                    "bookLength": 284,
                                    "description": "This book provides an analysis of frequently used research techniques in animal ecology, identifying their limitations and misuses, as well as possible solutions to avoid such pitfalls. The contributors provide an overarching account of central theoretical and methodological controversies. The editors have forged comprehensive presentations of key topics in animal ecology, such as territory and home range estimates, habitation evaluation, population viability analysis, GIS mapping, and measuring the dynamics of societies.",
                                    "summary_reviews": [],
                                    "avg_score": None
                                }
                            ]
                        },
                        "response": {
                            "raw_output": "{\"title\":\"How Animals Live (The Amazing World of Animals in the Wild)\"}"
                        }
                    },
                    {
                        "module": "BookBuyAgentRunner",
                        "prompt": {
                            "system": "\n    You are a ReAct BookBuy agent.\n\n    You have tools:\n    - recommendationTool(user_prompt, excluded_titles, user_preferences) -> dict\n    - findPricesTool(book_title) -> dict (returns ALL offers)\n    - buyBookTool(shop_id, book_title, address, payment_token) -> dict\n\n    You have EXACTLY 3 attempts total.\n    An attempt means: pick ONE candidate book and try to complete the whole process:\n    recommendationTool -> findPricesTool -> buyBookTool.\n\n    Attempt rules (within ONE attempt):\n    1) Call recommendationTool using the excluded_titles and user_preferences given in CONTEXT.\n    2) If recommendationTool returns status=\"no_match\": STOP ENTIRE RUN immediately.\n    3) Call findPricesTool with the recommendation.title .\n    4) If findPricesTool returns status=\"out_of_stock\" or status=\"error\": STOP this attempt immediately.\n    5) If findPricesTool returns status=\"found\":\n       - consider only offers with in_stock=true and price not None\n       - choose the lowest price\n       - If the lowest price is around or above 200 ILS, consider it expensive for a book and stop this attempt; otherwise proceed to purchase.\n       - when calling buyBookTool, prefer offer.store_title if present, else use the recommended title\n    6) Call buyBookTool exactly once.\n    7) If buyBookTool returns status=\"success\": you are DONE (final success).\n    8) If buyBookTool returns status=\"failed\": STOP this attempt immediately.\n\n    IMPORTANT:\n    - Do NOT retry within the same attempt. If something fails, stop the attempt.\n\n    CONTEXT:\n    attempt_number: 1 / 3\n    excluded_titles: [\"The Jungle Book\", \"Life of Pi\"]\n    user_preferences: [\"animals\", \"wildlife\", \"nature\", \"animal behavior\", \"book length: around 150-300 pages\"]\n    address: Tel Aviv\n    payment_token: 9876543\n    ",
                            "user": "I'm looking for an interesting book about wildlife that explains how animals survive and interact in nature",
                            "last_tool_result": {
                                "tool_name": "recommendationTool",
                                "args": {
                                    "user_prompt": "I'm looking for an interesting book about wildlife that explains how animals survive and interact in nature",
                                    "excluded_titles": [
                                        "The Jungle Book",
                                        "Life of Pi"
                                    ],
                                    "user_preferences": [
                                        "animals",
                                        "wildlife",
                                        "nature",
                                        "animal behavior",
                                        "book length: around 150-300 pages"
                                    ]
                                },
                                "result": {
                                    "status": "found",
                                    "title": "How Animals Live (The Amazing World of Animals in the Wild)",
                                    "authors": "Bernard Stonehouse, Esther Bertram",
                                    "published_date": "2004",
                                    "categories": "Juvenile Nonfiction",
                                    "book_length": 266,
                                    "description": "Presents an overview of the characteristic behavior, habits, and survival skills of various types of animals around the world."
                                }
                            }
                        },
                        "response": {
                            "content": "",
                            "tool_calls": [
                                {
                                    "name": "findPricesTool",
                                    "args": {
                                        "book_title": "How Animals Live (The Amazing World of Animals in the Wild)"
                                    },
                                    "id": "call_ocoTWDGwKqEJQb2NpW6rTgAM",
                                    "type": "tool_call"
                                }
                            ]
                        }
                    },
                    {
                        "module": "BookBuyAgentRunner",
                        "prompt": {
                            "system": "\n    You are a ReAct BookBuy agent.\n\n    You have tools:\n    - recommendationTool(user_prompt, excluded_titles, user_preferences) -> dict\n    - findPricesTool(book_title) -> dict (returns ALL offers)\n    - buyBookTool(shop_id, book_title, address, payment_token) -> dict\n\n    You have EXACTLY 3 attempts total.\n    An attempt means: pick ONE candidate book and try to complete the whole process:\n    recommendationTool -> findPricesTool -> buyBookTool.\n\n    Attempt rules (within ONE attempt):\n    1) Call recommendationTool using the excluded_titles and user_preferences given in CONTEXT.\n    2) If recommendationTool returns status=\"no_match\": STOP ENTIRE RUN immediately.\n    3) Call findPricesTool with the recommendation.title .\n    4) If findPricesTool returns status=\"out_of_stock\" or status=\"error\": STOP this attempt immediately.\n    5) If findPricesTool returns status=\"found\":\n       - consider only offers with in_stock=true and price not None\n       - choose the lowest price\n       - If the lowest price is around or above 200 ILS, consider it expensive for a book and stop this attempt; otherwise proceed to purchase.\n       - when calling buyBookTool, prefer offer.store_title if present, else use the recommended title\n    6) Call buyBookTool exactly once.\n    7) If buyBookTool returns status=\"success\": you are DONE (final success).\n    8) If buyBookTool returns status=\"failed\": STOP this attempt immediately.\n\n    IMPORTANT:\n    - Do NOT retry within the same attempt. If something fails, stop the attempt.\n\n    CONTEXT:\n    attempt_number: 1 / 3\n    excluded_titles: [\"The Jungle Book\", \"Life of Pi\"]\n    user_preferences: [\"animals\", \"wildlife\", \"nature\", \"animal behavior\", \"book length: around 150-300 pages\"]\n    address: Tel Aviv\n    payment_token: 9876543\n    ",
                            "user": "I'm looking for an interesting book about wildlife that explains how animals survive and interact in nature",
                            "last_tool_result": {
                                "tool_name": "findPricesTool",
                                "args": {
                                    "book_title": "How Animals Live (The Amazing World of Animals in the Wild)"
                                },
                                "result": {
                                    "status": "found",
                                    "title": "How Animals Live (The Amazing World of Animals in the Wild)",
                                    "offers": [
                                        {
                                            "shop": "fiction_boutique",
                                            "price": 52.84,
                                            "in_stock": True,
                                            "store_title": "How Animals Live (The Amazing World of Animals in the Wild)"
                                        }
                                    ],
                                    "errors": [
                                        {
                                            "shop": "knowledge_store",
                                            "error": "HTTP 404: {\"detail\":\"Book not found in this shop\"}"
                                        },
                                        {
                                            "shop": "mega_market1",
                                            "error": "HTTP 404: {\"detail\":\"Book not found in this shop\"}"
                                        },
                                        {
                                            "shop": "mega_market2",
                                            "error": "HTTP 404: {\"detail\":\"Book not found in this shop\"}"
                                        }
                                    ]
                                }
                            }
                        },
                        "response": {
                            "content": "",
                            "tool_calls": [
                                {
                                    "name": "buyBookTool",
                                    "args": {
                                        "shop_id": "fiction_boutique",
                                        "book_title": "How Animals Live (The Amazing World of Animals in the Wild)",
                                        "address": "Tel Aviv",
                                        "payment_token": "9876543"
                                    },
                                    "id": "call_P3mesEDeRrVkL7M0w5tDA81g",
                                    "type": "tool_call"
                                }
                            ]
                        }
                    }
                ]
            },
            {
                "prompt": {
                    "prompt": "I'm looking for an interesting book about tali daniel and offek",
                    "address": "Tel Aviv",
                    "payment_token": "9876543",
                    "user_preferences": [],
                    "disliked_titles": [],
                    "already_read_titles": []
                },
                "status": "ok",
                "error": None,
                "response": "Sorry — I couldn't find any suitable recommendation for your request, so I didn’t proceed to purchase attempts. Try broadening the topic or updating your preferences, and I’ll try again.",
                "steps": [
                    {
                        "module": "BookBuyAgentRunner",
                        "prompt": {
                            "system": "\n    You are a ReAct BookBuy agent.\n\n    You have tools:\n    - recommendationTool(user_prompt, excluded_titles, user_preferences) -> dict\n    - findPricesTool(book_title) -> dict (returns ALL offers)\n    - buyBookTool(shop_id, book_title, address, payment_token) -> dict\n\n    You have EXACTLY 3 attempts total.\n    An attempt means: pick ONE candidate book and try to complete the whole process:\n    recommendationTool -> findPricesTool -> buyBookTool.\n\n    Attempt rules (within ONE attempt):\n    1) Call recommendationTool using the excluded_titles and user_preferences given in CONTEXT.\n    2) If recommendationTool returns status=\"no_match\": STOP ENTIRE RUN immediately.\n    3) Call findPricesTool with the recommendation.title .\n    4) If findPricesTool returns status=\"out_of_stock\" or status=\"error\": STOP this attempt immediately.\n    5) If findPricesTool returns status=\"found\":\n       - consider only offers with in_stock=true and price not None\n       - choose the lowest price\n       - If the lowest price is around or above 200 ILS, consider it expensive for a book and stop this attempt; otherwise proceed to purchase.\n       - when calling buyBookTool, prefer offer.store_title if present, else use the recommended title\n    6) Call buyBookTool exactly once.\n    7) If buyBookTool returns status=\"success\": you are DONE (final success).\n    8) If buyBookTool returns status=\"failed\": STOP this attempt immediately.\n\n    IMPORTANT:\n    - Do NOT retry within the same attempt. If something fails, stop the attempt.\n\n    CONTEXT:\n    attempt_number: 1 / 3\n    excluded_titles: []\n    user_preferences: []\n    address: Tel Aviv\n    payment_token: 9876543\n    ",
                            "user": "I'm looking for an interesting book about tali daniel and offek"
                        },
                        "response": {
                            "content": "",
                            "tool_calls": [
                                {
                                    "name": "recommendationTool",
                                    "args": {
                                        "user_prompt": "I'm looking for an interesting book about tali daniel and offek",
                                        "excluded_titles": [],
                                        "user_preferences": []
                                    },
                                    "id": "call_kmz1x7gp7p57A9QqCnq9N8Qy",
                                    "type": "tool_call"
                                }
                            ]
                        }
                    },
                    {
                        "module": "DescriptionSelector",
                        "prompt": {
                            "user_prompt": "I'm looking for an interesting book about tali daniel and offek",
                            "user_preferences": [],
                            "candidate_books": [
                                {
                                    "title": "Tamar",
                                    "authors": "Mal Peet",
                                    "publishedDate": "2015-09-22",
                                    "categories": "Fiction",
                                    "bookLength": 396,
                                    "description": "From acclaimed British sensation Mal Peet comes a masterful story of adventure, love, secrets, and betrayal in time of war, both past and present. When her grandfather dies, Tamar inherits a box containing a series of clues and coded messages. Out of the past, another Tamar emerges, a man involved in the terrifying world of resistance fighters in Nazi-occupied Holland half a century before. His story is one of passionate love, jealousy, and tragedy set against the daily fear and casual horror of the Second World War -- and unraveling it is about to transform Tamar’s life forever."
                                },
                                {
                                    "title": "Tales Out of Shul: The Unorthodox Journal of an Orthodox Rabbi",
                                    "authors": "Emanuel Feldman",
                                    "publishedDate": "1996",
                                    "categories": "Social Science",
                                    "bookLength": 341,
                                    "description": "If there were a hall of fame of America's Orthodox rabbinate, Emanuel Feldman would be a charter member. Long before the word teshuvah became fashionable, he took a moribund congregation in Atlanta, turned it into a vibrant community, and led it for 40 years. In this poignant, delightful, provocative, uproarious, idealistic, uplifting journal, Rabbi Feldman takes us behind the pulpit as no one ever has before. Meet saints and scoundrels, righteous people and sinners, the movers and the meek. Tag along on countless everyday adventures. Taste sweet success and bitter failure. A marvelous book, by a heroic leader, graceful writer, and incisive thinker. Don't miss it! A Shaar Press Publication."
                                },
                                {
                                    "title": "The Devil's Shepherd: A Novel",
                                    "authors": "Steven Hartov",
                                    "publishedDate": "2013-03-04",
                                    "categories": "Fiction",
                                    "bookLength": 215,
                                    "description": "Israeli Military Intelligence agents Eytan Eckstein and Benni Baum are summoned once more to undertake a mission that could be their last. A defecting Czech spy claims to know the identity of a mole within Israel's top secret nuclear program, but he has fled to Africa and will only turn over the information if a string of his demands are met. Thus begins \"Operation Sorcerer,\" a quest to extricate the Czech spy, rescue a throng of desperate refugees, and survive the onslaught of Africa warlords determined to destroy the Israeli heroes. Steven Hartov was born in the United States and educated at Boston University. After serving in the U.S. Military Sealift Command, he emigrated to Israel and served in the Israel Defense Forces parachute corps and Military Intelligence special operations. He is the author of the espionage trilogy, \"The Heat of Ramadan,\" \"The Nylon Hand of God,\" and \"The Devil's Shepherd,\" and co-author of the New York Times best seller \"In the Company of Heroes\" and \"The Night Stalkers.\" For six years, Hartov helmed \"Special Operations Report\" as Editor-in-Chief. He currently serves as a Task Force commander in the New York Guard and is writing a new novel. Readers held in suspense throughout this galloping plot will end with their nails in their mouths, awaiting the next installment. -Kirkus Reviews A superb"
                                },
                                {
                                    "title": "Weirdo's war",
                                    "authors": "Michael Coleman",
                                    "publishedDate": "2008-10-01",
                                    "categories": "Adventure stories",
                                    "bookLength": 266,
                                    "description": "Daniel and Tosh are very different, so when they are put together on an Outward Bound course they are not happy. But when they fall in a cave with their teacher, who has broken his leg, they discover they each have skills that the other can draw on."
                                },
                                {
                                    "title": "The Sacrifice of Tamar",
                                    "authors": "Naomi Ragen",
                                    "publishedDate": "2010-07-20",
                                    "categories": "Fiction",
                                    "bookLength": 285,
                                    "description": "Tamar Finegold is twenty-one years old, the happy, beautiful bride of a rising young Rabbi in one of Brooklyn's insulated, ultra-Orthodox Jewish communities. Having married the man of her dreams and taken her place as a wife—and hopefully soon-to-be mother—in her community, Tamar feels as though the world is at her feet. But her secure, predictable existence is brought to an abrupt end when she is raped by an intruder. Fearing the unbearable stigma and threat to her marriage that could result from telling the truth, Tamar makes a fateful decision that changes her life forever. Her feeling that she did the only thing she could under the circumstances explodes when years later a shocking, undreamed of turn of events finally forces her to confront her past, once and for all"
                                },
                                {
                                    "title": "Passion, Betrayal and Killer Highlights",
                                    "authors": "Kyra Davis",
                                    "publishedDate": "2012-06-15",
                                    "categories": "Fiction",
                                    "bookLength": 219,
                                    "description": "Sophie Katz has just offered a man $12,000 for his services… Is she desperate or just meshugeneh? Considering the kind of disasters that usually befall the half-black, half-Jewish mystery writer, probably both. Because the last time Sophie saw sexy P.I. Anatoly Darinsky, he practically danced a jig when she waved goodbye &151; a normal reaction for a man who'd nearly bought the farm trying to protect her from her own foolishness. What are the chances he'd agree to take incriminating pictures of her sister's philandering husband? Or that he'd let her tag along &151; you know…for research? But when her brother-in-law turns up dead and her sister becomes the prime suspect, Sophie's priority is finding the real killer. With or without Anatoly's help. Her brother-in-law's secret life yields plenty of suspects, but the San Francisco police aren't taking any of them seriously. So Sophie does what comes naturally to her: she stirs up trouble (to lure the killer out, of course). But if her crazy plan works, will Anatoly be there to protect her this time?"
                                },
                                {
                                    "title": "The Tamarind Seed",
                                    "authors": "Evelyn Anthony",
                                    "publishedDate": "2015-11-17",
                                    "categories": "Fiction",
                                    "bookLength": 295,
                                    "description": "An Englishwoman falls for a Russian wanted by Intelligence on both sides of the Iron Curtain in this classic tale of Cold War espionage As executive assistant to a senior diplomat at the UN, widow Judith Farrow spends most of her working hours handling classified information. When her boss insists she take some time off in Barbados, she’s happy to escape her dead-end love affair with a very prominent, very married British attaché. But from the moment Judith meets Feodor Sverdlov, her low-key vacation turns into an international nightmare that threatens her job—and her life. A disillusioned military attaché working for the Soviet Embassy in Washington, DC, Sverdlov is known as a very dangerous man east and west of the Iron Curtain. Neither the British SIS nor the CIA believes his trip to the West Indies was an accident of fate. Suddenly Judith is perceived as a high-level risk, and Intelligence agent Jack Loder is dispatched to neutralize the situation. Now, Judith and Loder must identify the traitor in their midst—a mole code-named “Blue,” who’s firmly entrenched in DC’s power circles and preparing to deliver an irreversible blow to western civilization—before it’s too late."
                                }
                            ]
                        },
                        "response": {
                            "raw_output": ""
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
            user_preferences=request.user_preferences,
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
            temperature=1,
        )

        runner = BookBuyAgentRunner(llm, user)
        result = runner.run(request.prompt)

        return {
            "status": result.get("status"),
            "error": None,
            "response": result.get("response"),
            "steps": result.get("steps", []),
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
