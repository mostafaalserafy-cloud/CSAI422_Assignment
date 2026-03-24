# Assignment 3 — Weather assistant (tools + reasoning)

This is a lightweight command-line helper that responds to weather queries using live information. It communicates with an LLM through function calling and can also use a calculator for comparisons. There is a basic mode, a “step-by-step thinking” mode, and an advanced mode that includes safer tool usage and parallel executions when suitable.

## How to run it

**1.** Open this directory and (optionally) create a virtual environment:

```bash
cd "Assignment 3"
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

**2.** Add your API credentials:

- `API_KEY`, `BASE_URL`, `LLM_MODEL` — provided by your course or any OpenAI-compatible service.
- `WEATHER_API_KEY` — register at https://www.weatherapi.com/; the free plan is sufficient.

**3.** Launch the application:

```bash
python conversational_agent.py
```

A menu will appear: **1** = basic weather only, **2** = weather + calculator with clearer reasoning guidance, **3** = advanced (multi-step logic, parallel tool execution, optional JSON output). If you simply press Enter, it will prompt you again instead of making assumptions.

You can also bypass the menu and choose the mode directly from the terminal:

```bash
python conversational_agent.py 1
python conversational_agent.py --mode 2
```

Avoid committing the `.env` file or exposing real API keys on GitHub.

## Bonus mode (optional)

If you completed the bonus part, you can evaluate all three agents using a single query and store the ratings in a CSV file:

```bash
python conversational_agent.py --bonus "Compare the current weather in Cairo, Riyadh, and London."
```

## Code overview (brief)

- **Part 1:** Retrieves current weather and forecasts from WeatherAPI, connects them as tools, and runs a standard chat loop until tool usage is complete.
- **Part 2:** Introduces a calculator tool along with a system prompt encouraging stepwise reasoning before answering.
- **Part 3:** Implements safer tool handling, helpers for sequential vs parallel timing, a loop supporting multiple tool interactions per question, and utilities to validate a final JSON response when structured output is needed.

The advanced agent does not enforce structured JSON responses for every reply by default — simple queries may not match that format. If structured output is required for reporting, enable it in the code (`emit_structured_output=True` in `run_conversation_advanced`) or manually call `get_structured_final_response` after a longer dialogue.

## Prompt ideas

Start with a quick single-line question in mode 1, then try something requiring calculations in mode 2, and finally a multi-city or multi-step query in mode 3. Examples:

- “What’s the weather in Tokyo right now?”
- “Which is warmer in °C: 20°C in Paris or 68°F in New York?”
- “What’s the temperature difference between Cairo and London right now?”
- “Average maximum temperature in Cairo for the next 3 days?”
- “Compare the weather in Cairo, Riyadh, and London.”

## Notes for the report

Basic mode is the simplest, using only weather tools. Mode 2 is helpful when numerical calculations are required after retrieving weather data. Mode 3 adds tool-related error handling, parallel execution for independent calls, and supports multiple tool exchanges before producing a final answer.

If issues occur, they are usually due to incorrect API keys, an invalid base URL, or a location name not recognized by the weather service. For JSON validation, choose a comparison-type or calculation-based query so the model has enough information to populate every field.

## Helpful references

- OpenAI function calling
- JSON / structured outputs
- WeatherAPI documentation
