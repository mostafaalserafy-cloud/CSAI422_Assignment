"""
CSAI 422 Assignment 3 — Conversational weather assistant with tool use and reasoning.
"""

import csv
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor

import requests
from dotenv import load_dotenv
from openai import OpenAI

# Load environment variables from .env file
load_dotenv()

API_KEY = os.environ.get("API_KEY", os.getenv("OPTOGPT_API_KEY"))
BASE_URL = os.environ.get("BASE_URL", os.getenv("BASE_URL"))
LLM_MODEL = os.environ.get("LLM_MODEL", os.getenv("OPTOGPT_MODEL"))

# Initialize the OpenAI client with custom base URL
client = OpenAI(
    api_key=API_KEY,
    base_url=BASE_URL,
)

MAX_TOOL_ROUNDS_BASIC = 10


def _message_to_dict(msg):
    """Normalize SDK message objects to plain dicts for the conversation list."""
    if isinstance(msg, dict):
        return msg
    if hasattr(msg, "model_dump"):
        return msg.model_dump(exclude_none=True)
    return {"role": msg.role, "content": msg.content}


# --- Part 1: Weather tools ----------------------------------------------------


def get_current_weather(location):
    """Get the current weather for a location."""
    api_key = os.environ.get("WEATHER_API_KEY")
    url = (
        f"http://api.weatherapi.com/v1/current.json"
        f"?key={api_key}&q={location}&aqi=no"
    )
    response = requests.get(url, timeout=30)
    data = response.json()
    if "error" in data:
        return f"Error: {data['error']['message']}"
    weather_info = data["current"]
    return json.dumps(
        {
            "location": data["location"]["name"],
            "temperature_c": weather_info["temp_c"],
            "temperature_f": weather_info["temp_f"],
            "condition": weather_info["condition"]["text"],
            "humidity": weather_info["humidity"],
            "wind_kph": weather_info["wind_kph"],
        }
    )


def get_weather_forecast(location, days=3):
    """Get a weather forecast for a location for a specified number of days."""
    api_key = os.environ.get("WEATHER_API_KEY")
    url = (
        f"http://api.weatherapi.com/v1/forecast.json"
        f"?key={api_key}&q={location}&days={days}&aqi=no"
    )
    response = requests.get(url, timeout=30)
    data = response.json()
    if "error" in data:
        return f"Error: {data['error']['message']}"
    forecast_days = data["forecast"]["forecastday"]
    forecast_data = []
    for day in forecast_days:
        forecast_data.append(
            {
                "date": day["date"],
                "max_temp_c": day["day"]["maxtemp_c"],
                "min_temp_c": day["day"]["mintemp_c"],
                "condition": day["day"]["condition"]["text"],
                "chance_of_rain": day["day"]["daily_chance_of_rain"],
            }
        )
    return json.dumps(
        {
            "location": data["location"]["name"],
            "forecast": forecast_data,
        }
    )


weather_tools = [
    {
        "type": "function",
        "function": {
            "name": "get_current_weather",
            "description": "Get the current weather in a given location",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": (
                            "The city and state, e.g., San Francisco, CA, "
                            "or a country, e.g., France"
                        ),
                    }
                },
                "required": ["location"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_weather_forecast",
            "description": (
                "Get the weather forecast for a location for a specific "
                "number of days"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": (
                            "The city and state, e.g., San Francisco, CA, "
                            "or a country, e.g., France"
                        ),
                    },
                    "days": {
                        "type": "integer",
                        "description": "The number of days to forecast (1-10)",
                        "minimum": 1,
                        "maximum": 10,
                    },
                },
                "required": ["location"],
            },
        },
    },
]

available_functions = {
    "get_current_weather": get_current_weather,
    "get_weather_forecast": get_weather_forecast,
}


def process_messages(client, messages, tools=None, available_functions=None):
    """
    Process messages and invoke tools as needed until the model returns text
    (no further tool calls) or a safety limit is reached.
    """
    tools = tools or []
    available_functions = available_functions or {}
    rounds = 0
    while rounds < MAX_TOOL_ROUNDS_BASIC:
        rounds += 1
        response = client.chat.completions.create(
            model=LLM_MODEL,
            messages=messages,
            tools=tools,
        )
        response_message = response.choices[0].message
        messages.append(_message_to_dict(response_message))

        if not response_message.tool_calls:
            break

        for tool_call in response_message.tool_calls:
            function_name = tool_call.function.name
            function_to_call = available_functions[function_name]
            function_args = json.loads(tool_call.function.arguments)
            function_response = function_to_call(**function_args)
            messages.append(
                {
                    "tool_call_id": tool_call.id,
                    "role": "tool",
                    "name": function_name,
                    "content": function_response,
                }
            )

    return messages


def run_conversation(client, system_message="You are a helpful weather assistant."):
    """
    Run a conversation with the user, processing their messages and handling
    tool calls.
    """
    messages = [
        {
            "role": "system",
            "content": system_message,
        }
    ]
    print("Weather Assistant: Hello! I can help you with weather information.")
    print("Ask me about the weather anywhere!")
    print("(Type 'exit' to end the conversation)\n")

    while True:
        user_input = input("You: ")
        if user_input.lower() in ["exit", "quit", "bye"]:
            print("\nWeather Assistant: Goodbye! Have a great day!")
            break
        messages.append(
            {
                "role": "user",
                "content": user_input,
            }
        )
        messages = process_messages(
            client,
            messages,
            weather_tools,
            available_functions,
        )
        last_message = messages[-1]
        if last_message["role"] == "assistant" and last_message.get("content"):
            print(f"\nWeather Assistant: {last_message['content']}\n")

    return messages


# --- Part 2: Chain of Thought + calculator ------------------------------------


def calculator(expression):
    """
    Evaluate a mathematical expression.
    Note: eval is not safe for production use; acceptable for this assignment.
    """
    try:
        result = eval(expression)
        return str(result)
    except Exception as e:
        return f"Error: {str(e)}"


calculator_tool = {
    "type": "function",
    "function": {
        "name": "calculator",
        "description": "Evaluate a mathematical expression",
        "parameters": {
            "type": "object",
            "properties": {
                "expression": {
                    "type": "string",
                    "description": (
                        "The mathematical expression to evaluate, "
                        "e.g., '2 + 2' or '5 * (3 + 2)'"
                    ),
                }
            },
            "required": ["expression"],
        },
    },
}

cot_tools = weather_tools + [calculator_tool]
available_functions["calculator"] = calculator

cot_system_message = """You are a helpful assistant that can answer questions
about weather and perform calculations.
When responding to complex questions, please follow these steps:
1. Think step-by-step about what information you need.
2. Break down the problem into smaller parts.
3. Use the appropriate tools to gather information.
4. Explain your reasoning clearly.
5. Provide a clear final answer.
For example, if someone asks about temperature conversions or
comparisons between cities, first get the weather data, then use the
calculator if needed, showing your work.
"""


# --- Part 3: Safe execution, parallel tools, advanced workflow ----------------


def execute_tool_safely(tool_call, available_functions):
    """
    Execute a tool call with validation and error handling.
    Returns a JSON string describing either a success result or an error.
    """
    function_name = tool_call.function.name
    if function_name not in available_functions:
        return json.dumps(
            {
                "success": False,
                "error": f"Unknown function: {function_name}",
            }
        )
    try:
        function_args = json.loads(tool_call.function.arguments)
    except json.JSONDecodeError as e:
        return json.dumps(
            {
                "success": False,
                "error": f"Invalid JSON arguments: {str(e)}",
            }
        )
    try:
        function_response = available_functions[function_name](**function_args)
        return json.dumps(
            {
                "success": True,
                "function_name": function_name,
                "result": function_response,
            }
        )
    except TypeError as e:
        return json.dumps(
            {
                "success": False,
                "error": f"Invalid arguments: {str(e)}",
            }
        )
    except Exception as e:
        return json.dumps(
            {
                "success": False,
                "error": f"Tool execution failed: {str(e)}",
            }
        )


def execute_tools_sequential(tool_calls, available_functions):
    """
    Execute tool calls one after another.
    """
    results = []
    for tool_call in tool_calls:
        safe_result = execute_tool_safely(tool_call, available_functions)
        tool_message = {
            "tool_call_id": tool_call.id,
            "role": "tool",
            "name": tool_call.function.name,
            "content": safe_result,
        }
        results.append(tool_message)
    return results


def execute_tools_parallel(tool_calls, available_functions, max_workers=4):
    """Execute independent tool calls in parallel."""

    def run_single_tool(tool_call):
        return {
            "tool_call_id": tool_call.id,
            "role": "tool",
            "name": tool_call.function.name,
            "content": execute_tool_safely(tool_call, available_functions),
        }

    if not tool_calls:
        return []
    with ThreadPoolExecutor(
        max_workers=min(max_workers, len(tool_calls))
    ) as executor:
        return list(executor.map(run_single_tool, tool_calls))


def compare_parallel_vs_sequential(tool_calls, available_functions):
    """
    Measure the timing difference between sequential and parallel execution.
    """
    start = time.perf_counter()
    sequential_results = execute_tools_sequential(tool_calls, available_functions)
    sequential_time = time.perf_counter() - start

    start = time.perf_counter()
    parallel_results = execute_tools_parallel(tool_calls, available_functions)
    parallel_time = time.perf_counter() - start

    speedup = sequential_time / parallel_time if parallel_time > 0 else None
    return {
        "sequential_results": sequential_results,
        "parallel_results": parallel_results,
        "sequential_time": sequential_time,
        "parallel_time": parallel_time,
        "speedup": speedup,
    }


advanced_tools = cot_tools
advanced_system_message = """You are a helpful weather assistant that can use
weather tools and a calculator to solve multi-step problems.
Guidelines:
1. If the user asks about several independent locations, use multiple weather
   tool calls in parallel when appropriate.
2. If a question requires several steps, continue using tools until the task is
   completed.
3. If a tool fails, explain the issue clearly and continue safely when possible.
4. For complex comparison or calculation queries, prepare a structured final
   response.
"""


def process_messages_advanced(client, messages, tools=None, available_functions=None):
    """Send messages to the model and execute any returned tools in parallel."""
    tools = tools or []
    available_functions = available_functions or {}
    response = client.chat.completions.create(
        model=LLM_MODEL,
        messages=messages,
        tools=tools,
    )
    response_message = response.choices[0].message
    messages.append(_message_to_dict(response_message))
    if response_message.tool_calls:
        tool_results = execute_tools_parallel(
            response_message.tool_calls,
            available_functions,
        )
        messages.extend(tool_results)
    return messages, response_message


required_output_keys = [
    "query_type",
    "locations",
    "summary",
    "tool_calls_used",
    "final_answer",
]

structured_output_prompt = """For complex comparison or calculation queries,
return the final answer as a valid JSON object with exactly these keys:
- query_type
- locations
- summary
- tool_calls_used
- final_answer
Do not include markdown fences.
"""


def validate_structured_output(response_text):
    """Validate the final structured JSON response."""
    try:
        parsed = json.loads(response_text)
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON output: {str(e)}") from e
    for key in required_output_keys:
        if key not in parsed:
            raise ValueError(f"Missing required key: {key}")
    if not isinstance(parsed["locations"], list):
        raise ValueError("'locations' must be a list")
    if not isinstance(parsed["tool_calls_used"], list):
        raise ValueError("'tool_calls_used' must be a list")
    return parsed


def get_structured_final_response(client, messages):
    """
    Request a structured final response in JSON mode and validate it.
    """
    structured_messages = messages + [
        {
            "role": "system",
            "content": structured_output_prompt,
        }
    ]
    response = client.chat.completions.create(
        model=LLM_MODEL,
        messages=structured_messages,
        response_format={"type": "json_object"},
    )
    content = response.choices[0].message.content
    return validate_structured_output(content)


def run_conversation_advanced(
    client,
    system_message=None,
    max_iterations=5,
    emit_structured_output=False,
):
    """
    Run a conversation that supports multi-step tool workflows.
    """
    if system_message is None:
        system_message = advanced_system_message
    messages = [
        {
            "role": "system",
            "content": system_message,
        }
    ]
    print("Advanced Weather Assistant: Hello! Ask me complex weather questions.")
    print("I can compare cities, perform calculations, and return structured outputs.")
    print("(Type 'exit' to end the conversation)\n")

    while True:
        user_input = input("You: ")
        if user_input.lower() in ["exit", "quit", "bye"]:
            print("\nAdvanced Weather Assistant: Goodbye! Have a great day!")
            break
        messages.append(
            {
                "role": "user",
                "content": user_input,
            }
        )
        for _ in range(max_iterations):
            messages, response_message = process_messages_advanced(
                client,
                messages,
                advanced_tools,
                available_functions,
            )
            if not response_message.tool_calls:
                text = response_message.content or ""
                if text.strip():
                    print(f"\nAdvanced Weather Assistant: {text}\n")
                if emit_structured_output:
                    try:
                        structured = get_structured_final_response(client, messages)
                        print(
                            "Structured output (validated JSON):\n"
                            f"{json.dumps(structured, indent=2)}\n"
                        )
                    except ValueError as e:
                        print(
                            f"(Structured output could not be validated: {e})\n"
                        )
                break
        else:
            print(
                "\nAdvanced Weather Assistant: I stopped after reaching the"
                " maximum number of tool iterations.\n"
            )

    return messages


# --- Bonus: comparative evaluation --------------------------------------------


def _single_turn_messages(system_message, user_query, tools):
    return [
        {"role": "system", "content": system_message},
        {"role": "user", "content": user_query},
    ]


def run_agent_turn(
    client,
    system_message,
    user_query,
    tools,
    funcs,
    use_safe_parallel=True,
    max_iterations=5,
):
    """
    Run one user query through the tool loop; returns assistant text, elapsed time,
    conversation messages, and (when multiple tools were used in one round) those
    tool calls for benchmarking parallel vs sequential execution.
    """
    messages = _single_turn_messages(system_message, user_query, tools)
    t0 = time.perf_counter()
    multi_tool_calls = None
    if use_safe_parallel:
        for _ in range(max_iterations):
            messages, response_message = process_messages_advanced(
                client, messages, tools, funcs
            )
            if response_message.tool_calls and len(response_message.tool_calls) > 1:
                multi_tool_calls = response_message.tool_calls
            if not response_message.tool_calls:
                break
    else:
        messages = process_messages(client, messages, tools, funcs)
    elapsed = time.perf_counter() - t0
    last = messages[-1]
    if last["role"] == "assistant" and last.get("content"):
        return last["content"], elapsed, messages, multi_tool_calls
    for m in reversed(messages):
        if m.get("role") == "assistant" and m.get("content"):
            return m["content"], elapsed, messages, multi_tool_calls
    return "", elapsed, messages, multi_tool_calls


def run_comparative_evaluation(
    client,
    user_query,
    csv_path="evaluation_results.csv",
):
    """
    Bonus: run the same query through Basic, CoT, and Advanced agents; optionally
    measure parallel vs sequential tool time; collect ratings; append to CSV.
    """
    results = {}
    timings = {}

    basic_funcs = {
        k: available_functions[k]
        for k in ("get_current_weather", "get_weather_forecast")
    }

    text, t, _, _ = run_agent_turn(
        client,
        "You are a helpful weather assistant.",
        user_query,
        weather_tools,
        basic_funcs,
        use_safe_parallel=False,
    )
    results["basic"] = text
    timings["basic_s"] = t

    text, t, _, _ = run_agent_turn(
        client,
        cot_system_message,
        user_query,
        cot_tools,
        available_functions,
        use_safe_parallel=False,
    )
    results["cot"] = text
    timings["cot_s"] = t

    text, t, _, multi_tool_calls = run_agent_turn(
        client,
        advanced_system_message,
        user_query,
        advanced_tools,
        available_functions,
        use_safe_parallel=True,
    )
    results["advanced"] = text
    timings["advanced_s"] = t

    parallel_info = None
    if multi_tool_calls:
        parallel_info = compare_parallel_vs_sequential(
            multi_tool_calls, available_functions
        )
        timings["sequential_tool_s"] = parallel_info["sequential_time"]
        timings["parallel_tool_s"] = parallel_info["parallel_time"]
        timings["tool_speedup"] = parallel_info["speedup"]

    print("\n--- Responses side by side ---\n")
    for name, body in results.items():
        print(f"### {name.upper()}\n{body}\n")

    if parallel_info:
        print(
            "Parallel vs sequential (tool execution only):\n"
            f"  sequential: {parallel_info['sequential_time']:.4f}s\n"
            f"  parallel:   {parallel_info['parallel_time']:.4f}s\n"
            f"  speedup:    {parallel_info['speedup']}\n"
        )

    ratings = {}
    for name in ("basic", "cot", "advanced"):
        r = input(f"Rate {name} response (1-5): ").strip()
        try:
            ratings[name] = int(r)
        except ValueError:
            ratings[name] = ""

    row = {
        "query": user_query,
        "basic_rating": ratings.get("basic", ""),
        "cot_rating": ratings.get("cot", ""),
        "advanced_rating": ratings.get("advanced", ""),
        "basic_time_s": timings.get("basic_s"),
        "cot_time_s": timings.get("cot_s"),
        "advanced_time_s": timings.get("advanced_s"),
        "sequential_tool_s": timings.get("sequential_tool_s", ""),
        "parallel_tool_s": timings.get("parallel_tool_s", ""),
        "tool_speedup": timings.get("tool_speedup", ""),
    }
    file_exists = os.path.isfile(csv_path)
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(row.keys()))
        if not file_exists:
            w.writeheader()
        w.writerow(row)
    print(f"\nSaved row to {csv_path}")


def _prompt_agent_choice():
    """Ask until the user enters 1, 2, or 3 (handles empty Enter and stray spaces)."""
    while True:
        raw = input(
            "Choose an agent type (1: Basic, 2: Chain of Thought, 3: Advanced): "
        ).strip()
        if raw in ("1", "2", "3"):
            return raw
        if raw == "":
            print("You need to type 1, 2, or 3 then press Enter.")
        else:
            print("Invalid choice. Type 1, 2, or 3.")


if __name__ == "__main__":
    import sys

    if len(sys.argv) >= 2 and sys.argv[1] == "--bonus":
        query = " ".join(sys.argv[2:]).strip() or input(
            "Enter a single test query (e.g. multi-city comparison): "
        )
        run_comparative_evaluation(client, query)
        sys.exit(0)

    # Skip the menu when run like: python conversational_agent.py 2
    if len(sys.argv) >= 2 and sys.argv[1] in ("1", "2", "3"):
        choice = sys.argv[1]
    elif len(sys.argv) >= 3 and sys.argv[1] == "--mode" and sys.argv[2] in ("1", "2", "3"):
        choice = sys.argv[2]
    else:
        choice = _prompt_agent_choice()

    if choice == "1":
        run_conversation(client, "You are a helpful weather assistant.")
    elif choice == "2":
        run_conversation(client, cot_system_message)
    elif choice == "3":
        run_conversation_advanced(client, advanced_system_message)
