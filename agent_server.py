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
                "prompt": "I'm looking for an interesting book about antal lykazky daniel shmulevich and offek hai",
                "address": "Technion, Haifa",
                "payment_token": "123456",
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
                "full_response": "Success! Bought 'How Animals Live (The Amazing World of Animals in the Wild)' from fiction_boutique (Txn: TXN-B76AA29E). Estimated delivery: 3-5 business days to Tel Aviv.",
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
                                        "user_prompt": "Find a single book about wildlife that explains how animals survive and interact in nature. Prefer books focused on animal behavior, ecology, and nature; length around 150-300 pages. Avoid excluded titles.",
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
                                    "id": "call_hhSJIuBJePpTDuO0CzsN3gDk",
                                    "type": "tool_call"
                                }
                            ]
                        }
                    },
                    {
                        "module": "DescriptionSelector",
                        "prompt": {
                            "user_prompt": "Find a single book about wildlife that explains how animals survive and interact in nature. Prefer books focused on animal behavior, ecology, and nature; length around 150-300 pages. Avoid excluded titles.",
                            "user_preferences": [
                                "animals",
                                "wildlife",
                                "nature",
                                "animal behavior",
                                "book length: around 150-300 pages"
                            ],
                            "candidate_books": [
                                {
                                    "title": "Wildlife Biology",
                                    "authors": "Raymond Fredric Dasmann, Raymond F. Dasmann",
                                    "publishedDate": "1981-01-19",
                                    "categories": "Nature",
                                    "bookLength": 339,
                                    "description": "What good are wild animals?; Wildlife values; Wildlife as a natural resource; The status of wildlife conservation; The ten-thousand year war; Everything tied together; Functioning of ecosystems; Ecosystem development; Change and degradation; Distribution of biotic communities; Ecologic niches; A place to live; Enough to eat; Cover; Water; Quantity versus interspresion; Limiting factors; Introducing wildlife populations; Density and biomass; Population structure; Natality; Mortality; Interaction of population; Characteristics; Turnover; Productivity; Territory and travels; Movements internal to the population area; Movements external to the population; Group size and spacing; Territory; Significance of spacing and movements; Too many mice, too few elephants; New populations in new habitats; Some real life problems; The annual cycle of populations; Shootable surplus; Stocking of game; Stability of populations; Evolutionary strategies; Levelling off; The many meanings of carrying capacity; Declining diversity; Species extinction and area size; Destruction of habitat; Nature reserves; Animal trade; International assistance; The controllers.",
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
                                },
                                {
                                    "title": "Illustrated Nature Encyclopedia: Animals in the Wild (Illustrated Wildlife Encyclopedia)",
                                    "authors": "Tom Jackson",
                                    "publishedDate": "2009",
                                    "categories": "Animals",
                                    "bookLength": 341,
                                    "description": "This comprehensive guide provides information about North American habitats and the creatures found in them. It details the physiology and anatomy of each species, together with information on their ecology and evolution, their senses, survival and reproductive habitats."
                                },
                                {
                                    "title": "Dinnertime for animals (Books for young explorers)",
                                    "authors": "Jane R. McCauley",
                                    "publishedDate": "1991",
                                    "categories": "Animals",
                                    "bookLength": 277,
                                    "description": "Explains how different animals living in the wild have divergent diets, eating habits, and methods of gathering food for survival."
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
                                    "title": "Wildlife in America's History",
                                    "authors": "Peter Matthiessen",
                                    "publishedDate": "1977",
                                    "categories": "Animals",
                                    "bookLength": 729,
                                    "description": "This classic history of the rare, threatened, and extinct animals of North America is a dramatic chronicle of man's role in the disappearance of great and small species of our land. \"Should be the number one source volume for everyone who embraces the philosophy of conservation\".--Roger Tory Peterson. Illustrations throughout."
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
                                }
                            ]
                        },
                        "response": {
                            "raw_output": "{ \"titles\": [\"Animal ecology (Biology series)\", \"Prey\", \"How Animals Live (The Amazing World of Animals in the Wild)\", \"Wildlife Biology\"] }"
                        }
                    },
                    {
                        "module": "ReviewFinalSelector",
                        "prompt": {
                            "user_prompt": "Find a single book about wildlife that explains how animals survive and interact in nature. Prefer books focused on animal behavior, ecology, and nature; length around 150-300 pages. Avoid excluded titles.",
                            "candidate_books_with_reviews": [
                                {
                                    "title": "Wildlife Biology",
                                    "authors": "Raymond Fredric Dasmann, Raymond F. Dasmann",
                                    "publishedDate": "1981-01-19",
                                    "categories": "Nature",
                                    "bookLength": 339,
                                    "description": "What good are wild animals?; Wildlife values; Wildlife as a natural resource; The status of wildlife conservation; The ten-thousand year war; Everything tied together; Functioning of ecosystems; Ecosystem development; Change and degradation; Distribution of biotic communities; Ecologic niches; A place to live; Enough to eat; Cover; Water; Quantity versus interspresion; Limiting factors; Introducing wildlife populations; Density and biomass; Population structure; Natality; Mortality; Interaction of population; Characteristics; Turnover; Productivity; Territory and travels; Movements internal to the population area; Movements external to the population; Group size and spacing; Territory; Significance of spacing and movements; Too many mice, too few elephants; New populations in new habitats; Some real life problems; The annual cycle of populations; Shootable surplus; Stocking of game; Stability of populations; Evolutionary strategies; Levelling off; The many meanings of carrying capacity; Declining diversity; Species extinction and area size; Destruction of habitat; Nature reserves; Animal trade; International assistance; The controllers.",
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
                                    "title": "How Animals Live (The Amazing World of Animals in the Wild)",
                                    "authors": "Bernard Stonehouse, Esther Bertram",
                                    "publishedDate": "2004",
                                    "categories": "Juvenile Nonfiction",
                                    "bookLength": 266,
                                    "description": "Presents an overview of the characteristic behavior, habits, and survival skills of various types of animals around the world.",
                                    "summary_reviews": [],
                                    "avg_score": None
                                }
                            ]
                        },
                        "response": {
                            "raw_output": "{\"title\":\"Prey\"}"
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
                                    "user_prompt": "Find a single book about wildlife that explains how animals survive and interact in nature. Prefer books focused on animal behavior, ecology, and nature; length around 150-300 pages. Avoid excluded titles.",
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
                                    "title": "Prey",
                                    "authors": "Michael Chinery",
                                    "published_date": "2000",
                                    "categories": "Predation (Biology)",
                                    "book_length": 246,
                                    "description": "Tropical rainforests cover only about 6% of the earth's surface, yet they are home to more than half the world's species. This book looks at the biology and behaviour of predators and prey describing how they live together in harmony and balance and how vital they are to the wellbeing of the whole world. It is one of a series of books which aims to provide an understanding of the unique ecosystems of the rainforest. It describes the amazing plants and animals and how they interact, and addresses local and global environmental and conservation issues."
                                }
                            }
                        },
                        "response": {
                            "content": "",
                            "tool_calls": [
                                {
                                    "name": "findPricesTool",
                                    "args": {
                                        "book_title": "Prey"
                                    },
                                    "id": "call_KAgfWIe4fDbiDekvOdSQR6Xo",
                                    "type": "tool_call"
                                }
                            ]
                        }
                    },
                    {
                        "module": "BookBuyAgentRunner",
                        "prompt": {
                            "system": "\n    You are a ReAct BookBuy agent.\n\n    You have tools:\n    - recommendationTool(user_prompt, excluded_titles, user_preferences) -> dict\n    - findPricesTool(book_title) -> dict (returns ALL offers)\n    - buyBookTool(shop_id, book_title, address, payment_token) -> dict\n\n    You have EXACTLY 3 attempts total.\n    An attempt means: pick ONE candidate book and try to complete the whole process:\n    recommendationTool -> findPricesTool -> buyBookTool.\n\n    Attempt rules (within ONE attempt):\n    1) Call recommendationTool using the excluded_titles and user_preferences given in CONTEXT.\n    2) If recommendationTool returns status=\"no_match\": STOP ENTIRE RUN immediately.\n    3) Call findPricesTool with the recommendation.title .\n    4) If findPricesTool returns status=\"out_of_stock\" or status=\"error\": STOP this attempt immediately.\n    5) If findPricesTool returns status=\"found\":\n       - consider only offers with in_stock=true and price not None\n       - choose the lowest price\n       - If the lowest price is around or above 200 ILS, consider it expensive for a book and stop this attempt; otherwise proceed to purchase.\n       - when calling buyBookTool, prefer offer.store_title if present, else use the recommended title\n    6) Call buyBookTool exactly once.\n    7) If buyBookTool returns status=\"success\": you are DONE (final success).\n    8) If buyBookTool returns status=\"failed\": STOP this attempt immediately.\n\n    IMPORTANT:\n    - Do NOT retry within the same attempt. If something fails, stop the attempt.\n\n    CONTEXT:\n    attempt_number: 2 / 3\n    excluded_titles: [\"The Jungle Book\", \"Life of Pi\", \"Prey\"]\n    user_preferences: [\"animals\", \"wildlife\", \"nature\", \"animal behavior\", \"book length: around 150-300 pages\"]\n    address: Tel Aviv\n    payment_token: 9876543\n    ",
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
                                            "Life of Pi",
                                            "Prey"
                                        ],
                                        "user_preferences": [
                                            "animals",
                                            "wildlife",
                                            "nature",
                                            "animal behavior",
                                            "book length: around 150-300 pages"
                                        ]
                                    },
                                    "id": "call_avxp1gSKmjapWgI7BlbK01OD",
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
                                    "description": "A comprehensive book about the controversial coyote includes sections on its life history, present status, effect of predation on other wildlife, and continuing adaptability to mankind",
                                    "summary_reviews": [],
                                    "avg_score": None
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
                                },
                                {
                                    "title": "Wild animals I have known and 200 drawings,",
                                    "authors": "Ernest Thompson, Ernest Thompson Seton",
                                    "publishedDate": "2017-11-16",
                                    "categories": "Biography & Autobiography",
                                    "bookLength": 252,
                                    "description": "'The most precious book of my childhood’ – Sir David Attenborough Meet Old Lobo, a gigantic grey wolf whose death-defying predations on sheep and cattle herds are the scourge of farmers and ranchers in the Currumpaw region of northern New Mexico. This great wolf is just one of the animals whose true stories come to life in this engrossing collection of tales by the celebrated naturalist Ernest Thompson Seton (1860-1946). Combining scientific observations of animals in their natural habitats with a romantic vision of nature and the narrative skills of a born storyteller, Seton created an extraordinary collection of tales that gave the animal story new force and believability as a literary genre. Critically and popularly acclaimed upon its initial appearance in 1898, Wild Animals I Have Known remains, more than a century later, the best-known and best-loved of his works. Each tale focuses on an individual creature: the clever crow, Silverspot; Raggylug, a young cottontail rabbit; the author's errant hound, Bingo; Redruff, a Don Valley partridge; a wild horse known as The Mustang; Vixen, The Springfield Fox; and Wully, faithful sheep dog by day and treacherous killer by night. Seton offers affectionate but realistic portraits of each animal,",
                                    "summary_reviews": [],
                                    "avg_score": None
                                }
                            ]
                        },
                        "response": {
                            "raw_output": "{ \"titles\": [\"How Animals Live (The Amazing World of Animals in the Wild)\", \"Animal ecology (Biology series)\", \"Wild animals I have known and 200 drawings,\", \"Coyotes: Predators and Survivors\"] }"
                        }
                    },
                    {
                        "module": "ReviewFinalSelector",
                        "prompt": {
                            "user_prompt": "I'm looking for an interesting book about wildlife that explains how animals survive and interact in nature",
                            "candidate_books_with_reviews": [
                                {
                                    "title": "Coyotes: Predators and Survivors",
                                    "authors": "Charles L. Cadieux",
                                    "publishedDate": "1983",
                                    "categories": "Science",
                                    "bookLength": 264,
                                    "description": "A comprehensive book about the controversial coyote includes sections on its life history, present status, effect of predation on other wildlife, and continuing adaptability to mankind",
                                    "summary_reviews": [],
                                    "avg_score": None
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
                                    "title": "Animal ecology (Biology series)",
                                    "authors": "Luigi Boitani, T. K. Fuller",
                                    "publishedDate": "2000-01",
                                    "categories": "Science",
                                    "bookLength": 284,
                                    "description": "This book provides an analysis of frequently used research techniques in animal ecology, identifying their limitations and misuses, as well as possible solutions to avoid such pitfalls. The contributors provide an overarching account of central theoretical and methodological controversies. The editors have forged comprehensive presentations of key topics in animal ecology, such as territory and home range estimates, habitation evaluation, population viability analysis, GIS mapping, and measuring the dynamics of societies.",
                                    "summary_reviews": [],
                                    "avg_score": None
                                },
                                {
                                    "title": "Wild animals I have known and 200 drawings,",
                                    "authors": "Ernest Thompson, Ernest Thompson Seton",
                                    "publishedDate": "2017-11-16",
                                    "categories": "Biography & Autobiography",
                                    "bookLength": 252,
                                    "description": "'The most precious book of my childhood’ – Sir David Attenborough Meet Old Lobo, a gigantic grey wolf whose death-defying predations on sheep and cattle herds are the scourge of farmers and ranchers in the Currumpaw region of northern New Mexico. This great wolf is just one of the animals whose true stories come to life in this engrossing collection of tales by the celebrated naturalist Ernest Thompson Seton (1860-1946). Combining scientific observations of animals in their natural habitats with a romantic vision of nature and the narrative skills of a born storyteller, Seton created an extraordinary collection of tales that gave the animal story new force and believability as a literary genre. Critically and popularly acclaimed upon its initial appearance in 1898, Wild Animals I Have Known remains, more than a century later, the best-known and best-loved of his works. Each tale focuses on an individual creature: the clever crow, Silverspot; Raggylug, a young cottontail rabbit; the author's errant hound, Bingo; Redruff, a Don Valley partridge; a wild horse known as The Mustang; Vixen, The Springfield Fox; and Wully, faithful sheep dog by day and treacherous killer by night. Seton offers affectionate but realistic portraits of each animal,",
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
                            "system": "\n    You are a ReAct BookBuy agent.\n\n    You have tools:\n    - recommendationTool(user_prompt, excluded_titles, user_preferences) -> dict\n    - findPricesTool(book_title) -> dict (returns ALL offers)\n    - buyBookTool(shop_id, book_title, address, payment_token) -> dict\n\n    You have EXACTLY 3 attempts total.\n    An attempt means: pick ONE candidate book and try to complete the whole process:\n    recommendationTool -> findPricesTool -> buyBookTool.\n\n    Attempt rules (within ONE attempt):\n    1) Call recommendationTool using the excluded_titles and user_preferences given in CONTEXT.\n    2) If recommendationTool returns status=\"no_match\": STOP ENTIRE RUN immediately.\n    3) Call findPricesTool with the recommendation.title .\n    4) If findPricesTool returns status=\"out_of_stock\" or status=\"error\": STOP this attempt immediately.\n    5) If findPricesTool returns status=\"found\":\n       - consider only offers with in_stock=true and price not None\n       - choose the lowest price\n       - If the lowest price is around or above 200 ILS, consider it expensive for a book and stop this attempt; otherwise proceed to purchase.\n       - when calling buyBookTool, prefer offer.store_title if present, else use the recommended title\n    6) Call buyBookTool exactly once.\n    7) If buyBookTool returns status=\"success\": you are DONE (final success).\n    8) If buyBookTool returns status=\"failed\": STOP this attempt immediately.\n\n    IMPORTANT:\n    - Do NOT retry within the same attempt. If something fails, stop the attempt.\n\n    CONTEXT:\n    attempt_number: 2 / 3\n    excluded_titles: [\"The Jungle Book\", \"Life of Pi\", \"Prey\"]\n    user_preferences: [\"animals\", \"wildlife\", \"nature\", \"animal behavior\", \"book length: around 150-300 pages\"]\n    address: Tel Aviv\n    payment_token: 9876543\n    ",
                            "user": "I'm looking for an interesting book about wildlife that explains how animals survive and interact in nature",
                            "last_tool_result": {
                                "tool_name": "recommendationTool",
                                "args": {
                                    "user_prompt": "I'm looking for an interesting book about wildlife that explains how animals survive and interact in nature",
                                    "excluded_titles": [
                                        "The Jungle Book",
                                        "Life of Pi",
                                        "Prey"
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
                                    "id": "call_fiZwOqjoTpZNa2hddg8zv2Ch",
                                    "type": "tool_call"
                                }
                            ]
                        }
                    },
                    {
                        "module": "BookBuyAgentRunner",
                        "prompt": {
                            "system": "\n    You are a ReAct BookBuy agent.\n\n    You have tools:\n    - recommendationTool(user_prompt, excluded_titles, user_preferences) -> dict\n    - findPricesTool(book_title) -> dict (returns ALL offers)\n    - buyBookTool(shop_id, book_title, address, payment_token) -> dict\n\n    You have EXACTLY 3 attempts total.\n    An attempt means: pick ONE candidate book and try to complete the whole process:\n    recommendationTool -> findPricesTool -> buyBookTool.\n\n    Attempt rules (within ONE attempt):\n    1) Call recommendationTool using the excluded_titles and user_preferences given in CONTEXT.\n    2) If recommendationTool returns status=\"no_match\": STOP ENTIRE RUN immediately.\n    3) Call findPricesTool with the recommendation.title .\n    4) If findPricesTool returns status=\"out_of_stock\" or status=\"error\": STOP this attempt immediately.\n    5) If findPricesTool returns status=\"found\":\n       - consider only offers with in_stock=true and price not None\n       - choose the lowest price\n       - If the lowest price is around or above 200 ILS, consider it expensive for a book and stop this attempt; otherwise proceed to purchase.\n       - when calling buyBookTool, prefer offer.store_title if present, else use the recommended title\n    6) Call buyBookTool exactly once.\n    7) If buyBookTool returns status=\"success\": you are DONE (final success).\n    8) If buyBookTool returns status=\"failed\": STOP this attempt immediately.\n\n    IMPORTANT:\n    - Do NOT retry within the same attempt. If something fails, stop the attempt.\n\n    CONTEXT:\n    attempt_number: 2 / 3\n    excluded_titles: [\"The Jungle Book\", \"Life of Pi\", \"Prey\"]\n    user_preferences: [\"animals\", \"wildlife\", \"nature\", \"animal behavior\", \"book length: around 150-300 pages\"]\n    address: Tel Aviv\n    payment_token: 9876543\n    ",
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
                                            "shop": "mega_market1",
                                            "error": "HTTP 404: {\"detail\":\"Book not found in this shop\"}"
                                        },
                                        {
                                            "shop": "knowledge_store",
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
                                    "id": "call_bBSub3zaSHn0UetT2neywxKN",
                                    "type": "tool_call"
                                }
                            ]
                        }
                    }
                ]
            },
            {
                "prompt": {
                    "prompt": "I'm looking for an interesting book about antal lykazky daniel shmulevich and offek hai",
                    "address": "Technion, Haifa",
                    "payment_token": "123456",
                    "user_preferences": [],
                    "disliked_titles": [],
                    "already_read_titles": []
                },
                "full_response": "Sorry — I couldn't find any suitable recommendation for your request, so I didn’t proceed to purchase attempts. Try broadening the topic or updating your preferences, and I’ll try again.",
                "steps": [
                    {
                        "module": "BookBuyAgentRunner",
                        "prompt": {
                            "system": "\n    You are a ReAct BookBuy agent.\n\n    You have tools:\n    - recommendationTool(user_prompt, excluded_titles, user_preferences) -> dict\n    - findPricesTool(book_title) -> dict (returns ALL offers)\n    - buyBookTool(shop_id, book_title, address, payment_token) -> dict\n\n    You have EXACTLY 3 attempts total.\n    An attempt means: pick ONE candidate book and try to complete the whole process:\n    recommendationTool -> findPricesTool -> buyBookTool.\n\n    Attempt rules (within ONE attempt):\n    1) Call recommendationTool using the excluded_titles and user_preferences given in CONTEXT.\n    2) If recommendationTool returns status=\"no_match\": STOP ENTIRE RUN immediately.\n    3) Call findPricesTool with the recommendation.title .\n    4) If findPricesTool returns status=\"out_of_stock\" or status=\"error\": STOP this attempt immediately.\n    5) If findPricesTool returns status=\"found\":\n       - consider only offers with in_stock=true and price not None\n       - choose the lowest price\n       - If the lowest price is around or above 200 ILS, consider it expensive for a book and stop this attempt; otherwise proceed to purchase.\n       - when calling buyBookTool, prefer offer.store_title if present, else use the recommended title\n    6) Call buyBookTool exactly once.\n    7) If buyBookTool returns status=\"success\": you are DONE (final success).\n    8) If buyBookTool returns status=\"failed\": STOP this attempt immediately.\n\n    IMPORTANT:\n    - Do NOT retry within the same attempt. If something fails, stop the attempt.\n\n    CONTEXT:\n    attempt_number: 1 / 3\n    excluded_titles: []\n    user_preferences: []\n    address: Technion, Haifa\n    payment_token: 123456\n    ",
                            "user": "I'm looking for an interesting book about antal lykazky daniel shmulevich and offek hai"
                        },
                        "response": {
                            "content": "",
                            "tool_calls": [
                                {
                                    "name": "recommendationTool",
                                    "args": {
                                        "user_prompt": "I'm looking for an interesting book about antal lykazky daniel shmulevich and offek hai",
                                        "excluded_titles": [],
                                        "user_preferences": []
                                    },
                                    "id": "call_MY1oXTL8KsAtYeEPcT1kuSFe",
                                    "type": "tool_call"
                                }
                            ]
                        }
                    },
                    {
                        "module": "DescriptionSelector",
                        "prompt": {
                            "user_prompt": "I'm looking for an interesting book about antal lykazky daniel shmulevich and offek hai",
                            "user_preferences": [],
                            "candidate_books": [
                                {
                                    "title": "Tales Out of Shul: The Unorthodox Journal of an Orthodox Rabbi",
                                    "authors": "Emanuel Feldman",
                                    "publishedDate": "1996",
                                    "categories": "Social Science",
                                    "bookLength": 341,
                                    "description": "If there were a hall of fame of America's Orthodox rabbinate, Emanuel Feldman would be a charter member. Long before the word teshuvah became fashionable, he took a moribund congregation in Atlanta, turned it into a vibrant community, and led it for 40 years. In this poignant, delightful, provocative, uproarious, idealistic, uplifting journal, Rabbi Feldman takes us behind the pulpit as no one ever has before. Meet saints and scoundrels, righteous people and sinners, the movers and the meek. Tag along on countless everyday adventures. Taste sweet success and bitter failure. A marvelous book, by a heroic leader, graceful writer, and incisive thinker. Don't miss it! A Shaar Press Publication."
                                },
                                {
                                    "title": "Uniter of Heaven and Earth: Rabbi Meshullam Feibush Heller of Zbarazh and the Rise of Hasidism in Eastern Galicia (Suny Series in Judaica) (Suny ... Hermeneutics, Mysticism and Religion)",
                                    "authors": "Miles Krassen",
                                    "publishedDate": "2012-02-01",
                                    "categories": "History",
                                    "bookLength": 151,
                                    "description": "A clear and penetrating account of the basis of Hasidic mysticism. Includes translations of many texts never before available in English."
                                },
                                {
                                    "title": "Rescued Year 1ST Edition Paperback",
                                    "authors": "Aaron Lansky",
                                    "publishedDate": "2005-09-02",
                                    "categories": "History",
                                    "bookLength": 335,
                                    "description": "account of his journey is both “extraordinary” (The Boston Globe) and “entertaining” (Los Angeles Times). “Lansky charmingly describes his adventures as president and founder of the National Yiddish Book Center, which now has new headquarters at Hampshire College in Amherst, Mass. To Lansky, Yiddish literature represented an important piece of Jewish cultural history, a link to the past and a memory of a generation lost to the Holocaust. Lansky’s account of salvaging books is both hilarious and moving, filled with Jewish humor, conversations with elderly Jewish immigrants for whom the books evoke memories of a faraway past, stories of desperate midnight rescues from rain-soaked dumpsters, and touching accounts of Lansky’s trips to what were once thriving Jewish communities in Europe. The book is a testimony to his love of Judaism and literature and his desire to make a difference in the world.” —Publishers Weekly"
                                },
                                {
                                    "title": "The Other New York Jewish Intellectuals (Reappraisals in Jewish Social and Intellectual History)",
                                    "authors": "Carole S Kessner",
                                    "publishedDate": "1994-10-01",
                                    "categories": "Biography & Autobiography",
                                    "bookLength": 347,
                                    "description": "Irving Howe. Saul Bellow. Lionel Trilling. These are names that immediately come to mind when one thinks of the New York Jewish intellectuals of the late thirties and forties. And yet the New York Jewish intellectual community was far larger and more diverse than is commonly thought. In The Other New York Jewish Intellectuals we find a group of thinkers who may not have had widespread celebrity status but who fostered a real sense of community within the Jewish world in these troubled times. What unified these men and women was their commitment and allegiance to the Jewish people. Here we find Hayim Greenberg, Henry Hurwitz, Marie Syrkin, Maurice Samuel, Ben Halperin, Trude Weiss-Rosmarin, Morris Raphael Cohen, Ludwig Lewisohn, Milton Steinberg, Will Herberg, A. M. Klein, and Mordecai Kaplan, and many others. Divided into 3 sections--Opinion Makers, Men of Letters, and Spiritual Leaders--the book will be of particular interest to students and others interested in Jewish studies, American intellectual history, as well as history of the 30s and 40s."
                                },
                                {
                                    "title": "Thirst: The Desert Trilogy",
                                    "authors": "Shulamith Hareven",
                                    "publishedDate": "2015-01-06",
                                    "categories": "Fiction",
                                    "bookLength": 221,
                                    "description": "the Academy of the Hebrew language. In 1962, she published her first book, a poetry volume titled Predatory Jerusalem. After that, she wrote and translated prose books and plays. She published essays and articles about Israeli society and culture in literary journals Masa, Orlogin, and Keshet, and in newspapers Al Ha-Mishmar, Maariv, and Yedioth Ahronoth. Her books have been translated into twenty-one languages. Hareven was an activist for Peace Now, and in 1995, the French weekly L'Express listed her among the 100 women \"who move the world.” Hillel Halkin is an American-born Israeli translator, biographer, literary critic, and novelist, who has lived in Israel since 1970. Halkin translates from Hebrew and Yiddish literature into English. He has translated Sholem Aleichem's Tevye the Dairyman, and major Hebrew and Israeli novelists, among them Yosef Haim Brenner, S. Y. Agnon, Shulamith Hareven, A. B. Yehoshua, Amos Oz, and Meir Shalev."
                                },
                                {
                                    "title": "The Last Kabbalist of Lisbon",
                                    "authors": "Richard Zimler",
                                    "publishedDate": "2000-03-15",
                                    "categories": "Fiction",
                                    "bookLength": 769,
                                    "description": "Just a few years earlier, Jews living in Portugal were dragged to the baptismal font and forced to convert to Christianity. Many of these New Christians persevered in their Jewish prayers and rituals in secret and at great risk; the hidden, arcane practices of the kabbalists, a mystical sect of Jews, continued as well. One such secret Jew was Berekiah Zarco, an intelligent young manuscript illuminator. Inflamed by love and revenge, he searches, in the crucible of the raging pogrom, for the killer of his beloved uncle Abraham, a renowned kabbalist and manuscript illuminator, discovered murdered in a hidden synagogue along with a young girl in dishabille. Risking his life in streets seething with mayhem, Berekiah tracks down answers among Christians, New Christians, Jews, and the fellow kabbalists of his uncle, whose secret language and codes by turns light and obscure the way to the truth he seeks. A marvelous story, a challenging mystery, and a telling tale of the evils of intolerance, The Last Kabbalist of Lisbon both compels and entertains."
                                },
                                {
                                    "title": "Selected Stories of Sholom Aleichem (The Modern Library of the World's Best Books)",
                                    "authors": "Sholem Aleichem",
                                    "publishedDate": "1956",
                                    "categories": "Jews",
                                    "bookLength": 386,
                                    "description": "(The Modern library of the World's best books [145])."
                                }
                            ]
                        },
                        "response": {
                            "raw_output": "{ \"titles\": [] }"
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
