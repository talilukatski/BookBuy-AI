import json
from typing import List, Optional, Any, Dict
from langchain_openai import ChatOpenAI
from langchain_core.messages import ToolMessage, AIMessage, HumanMessage, SystemMessage
from recommendation_tool import recommendation_tool
from find_and_buy_tools import find_prices, buy_book
from config import OPENAI_API_KEY, OPENAI_BASE_URL, LLM_MODEL
from user_personal_details import UserPersonalDetails
import find_and_buy_tools


class BookBuyAgentRunner:
    """
    Manages the ReAct workflow for finding and purchasing books.
    Handles up to 3 separate attempts to find a valid match.
    """
    def __init__(self, llm: ChatOpenAI, user: UserPersonalDetails):
        self.llm = llm
        self.user = user
        # Initialize tools
        self.tools = [recommendation_tool, find_prices, buy_book]
        # Bind tools natively to the LLM (OpenAI Tool Calling)
        self.llm_with_tools = self.llm.bind_tools(self.tools)

    def run(self, user_prompt: str) -> Dict[str, Any]:
        """
        Main execution loop for the agent.
        """
        excluded_titles = self.user.initial_excluded_titles()
        all_steps = []

        for attempt_number in range(1, 4):
            last_tool_result_for_trace = None

            system_context = f"""
    You are a ReAct BookBuy agent.

    You have tools:
    - recommendationTool(user_prompt, excluded_titles, user_preferences) -> dict
    - findPricesTool(book_title) -> dict (returns ALL offers)
    - buyBookTool(shop_id, book_title, address, payment_token) -> dict

    You have EXACTLY 3 attempts total.
    An attempt means: pick ONE candidate book and try to complete the whole process:
    recommendationTool -> findPricesTool -> buyBookTool.

    Attempt rules (within ONE attempt):
    1) Call recommendationTool using the excluded_titles and user_preferences given in CONTEXT.
    2) If recommendationTool returns status="no_match": STOP ENTIRE RUN immediately.
    3) Call findPricesTool with the recommendation.title .
    4) If findPricesTool returns status="out_of_stock" or status="error": STOP this attempt immediately.
    5) If findPricesTool returns status="found":
       - consider only offers with in_stock=true and price not null
       - choose the lowest price
       - If the lowest price is around or above 200 ILS, consider it expensive for a book and stop this attempt; otherwise proceed to purchase.
       - when calling buyBookTool, prefer offer.store_title if present, else use the recommended title
    6) Call buyBookTool exactly once.
    7) If buyBookTool returns status="success": you are DONE (final success).
    8) If buyBookTool returns status="failed": STOP this attempt immediately.

    IMPORTANT:
    - Do NOT retry within the same attempt. If something fails, stop the attempt.

    CONTEXT:
    attempt_number: {attempt_number} / 3
    excluded_titles: {json.dumps(excluded_titles, ensure_ascii=False)}
    user_preferences: {json.dumps(self.user.user_preferences, ensure_ascii=False)}
    address: {self.user.address}
    payment_token: {self.user.payment_token}
    """

            messages = [
                SystemMessage(content=system_context),
                HumanMessage(content=user_prompt)
            ]

            exit_current_attempt = False
            recommendation_called = False

            for step in range(8):
                runner_prompt = {
                    "system": system_context,
                    "user": user_prompt
                }
                if last_tool_result_for_trace is not None:
                    runner_prompt["last_tool_result"] = last_tool_result_for_trace

                ai_msg = self.llm_with_tools.invoke(messages)

                all_steps.append({
                    "module": "BookBuyAgentRunner",
                    "prompt": runner_prompt,
                    "response": {
                        "content": ai_msg.content,
                        "tool_calls": ai_msg.tool_calls or []
                    }
                })

                messages.append(ai_msg)

                if not ai_msg.tool_calls:
                    break

                for tool_call in ai_msg.tool_calls:
                    if tool_call["name"] == "recommendationTool" and recommendation_called:
                        observation = {
                            "status": "skipped",
                            "reason": "recommendationTool already called in this attempt"
                        }

                        last_tool_result_for_trace = {
                            "tool_name": tool_call["name"],
                            "args": tool_call["args"],
                            "result": observation
                        }

                        messages.append(
                            ToolMessage(
                                tool_call_id=tool_call["id"],
                                content=json.dumps(observation)
                            )
                        )
                        continue

                    if tool_call["name"] == "recommendationTool":
                        recommendation_called = True

                    selected_tool = {t.name: t for t in self.tools}[tool_call["name"]]
                    tool_args = tool_call["args"]
                    observation = selected_tool.invoke(tool_args)

                    if tool_call["name"] == "recommendationTool":
                        all_steps.extend(observation.get("llm_steps", []))

                    tool_message_payload = dict(observation)
                    tool_message_payload.pop("llm_steps", None)

                    last_tool_result_for_trace = {
                        "tool_name": tool_call["name"],
                        "args": tool_args,
                        "result": tool_message_payload
                    }

                    messages.append(
                        ToolMessage(
                            tool_call_id=tool_call["id"],
                            content=json.dumps(tool_message_payload)
                        )
                    )

                    if tool_call["name"] == "findPricesTool" and observation.get("status") == "found":
                        valid_offers = [
                            o for o in observation.get("offers", [])
                            if o.get("in_stock") and o.get("price") is not None
                        ]

                        if valid_offers:
                            best_offer = min(valid_offers, key=lambda o: o["price"])

                            if best_offer["price"] > 150:
                                title_to_exclude = observation.get("title") or tool_args.get("book_title")

                                if title_to_exclude and title_to_exclude not in excluded_titles:
                                    excluded_titles.append(title_to_exclude)

                                exit_current_attempt = True
                                break

                    is_purchase_successful = (
                            tool_call["name"] == "buyBookTool"
                            and observation.get("status") in ["success", "confirmed"]
                    )

                    if is_purchase_successful:
                        return {
                            "status": "ok",
                            "response": (
                                f"Success! Bought '{observation.get('title')}' from {observation.get('shop')} "
                                f"(Txn: {observation.get('transaction_id')}). "
                                f"Estimated delivery: {observation.get('eta')} to {self.user.address}."
                            ),
                            "steps": all_steps
                        }

                    if tool_call["name"] == "recommendationTool" and observation.get("status") == "no_match":
                        return {
                            "status": "fail",
                            "response": (
                                "Sorry — I couldn't find any suitable recommendation for your request, "
                                "so I didn’t proceed to purchase attempts. Try broadening the topic or "
                                "updating your preferences, and I’ll try again."
                            ),
                            "steps": all_steps
                        }

                    should_stop_this_attempt = (
                            (tool_call["name"] == "findPricesTool" and observation.get("status") in ["out_of_stock",
                                                                                                     "error"])
                            or
                            (tool_call["name"] == "buyBookTool" and observation.get("status") == "failed")
                    )

                    if should_stop_this_attempt:
                        title_to_exclude = observation.get("title") or tool_args.get("book_title")
                        if title_to_exclude and title_to_exclude not in excluded_titles:
                            excluded_titles.append(title_to_exclude)

                        exit_current_attempt = True
                        break

                if exit_current_attempt:
                    break

        return {
            "status": "fail",
            "response": (
                "I'm sorry, but I couldn't complete the purchase of a book that fits your "
                "preferences. It may be unavailable or out of stock at our partner shops. "
                "Please try again or adjust your request and I'll gladly help."
            ),
            "steps": all_steps
        }

if __name__ == "__main__":
    # ---- Manual smoke test for the ReAct agent runner ----
    # Make sure you already created `llm = ChatOpenAI(...)` somewhere above.

    # 1) Create a test user profile (preferences + already read/disliked)
    user = UserPersonalDetails(
        user_preferences=["astrology", "theoretical approach", "meditation exercises", "visual explanations",
            "book length: around 200-400 pages"],
        disliked_titles=["The Forbidden Stories of Marta Veneranda"],
        already_read_titles=["The House on Mango Street"],
        address="Daliya, Hifa",
        payment_token="1234567890"
    )

    llm = ChatOpenAI(
        model=LLM_MODEL,
        api_key=OPENAI_API_KEY,
        base_url=OPENAI_BASE_URL,
        max_tokens=1024,
        temperature=1,
    )

    # 2) Create the runner
    runner = BookBuyAgentRunner(llm, user)

    # 3) Pick a prompt that should clearly match books in your Pinecone index
    prompt = "I'm looking for a theoretical astrology book that explains astrological patterns using psychology, meditation exercises, and clear visual explanations."

    # 4) Run and print a clean summary
    result = runner.run(prompt)

    print("\n" + "=" * 80)
    print("FINAL STATUS:", result.get("status"))
    print("FINAL RESPONSE:", result.get("response"))
    print("=" * 80)

    # 5) Print steps per attempt so you can verify:
    #    attempt 1 -> recommendationTool -> findPricesTool -> (maybe) buyBookTool
    #    then attempt 2 / 3 if needed
    steps = result.get("steps", [])
    if not steps:
        print("No steps recorded (unexpected).")
    else:
        current_attempt = None
        for s in steps:
            if s["attempt"] != current_attempt:
                current_attempt = s["attempt"]
                print(f"\n--- ATTEMPT {current_attempt} ---")

            print(f"\nTOOL: {s['module']}")
            print("INPUT:", s["prompt"])
            print("OUTPUT:", s["response"])

    print("\nDone.\n")
