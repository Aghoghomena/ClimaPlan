# ClimaPlan

![Python](https://img.shields.io/badge/Python-3.13+-3776AB?style=flat&logo=python&logoColor=white)
![LangGraph](https://img.shields.io/badge/LangGraph-1.0+-1C3C3C?style=flat&logo=langchain&logoColor=white)
![Groq](https://img.shields.io/badge/LLM-Groq%20%7C%20Gemini-F55036?style=flat&logo=groq&logoColor=white)
![MCP](https://img.shields.io/badge/MCP-FastMCP-6C3483?style=flat)
![LangSmith](https://img.shields.io/badge/Observability-LangSmith-FF6B35?style=flat&logo=langchain&logoColor=white)
![Open-Meteo](https://img.shields.io/badge/Weather-Open--Meteo-00B4D8?style=flat)
![License](https://img.shields.io/badge/License-MIT-green?style=flat)
![Status](https://img.shields.io/badge/Status-In%20Development-yellow?style=flat)

> **A Person asks: "What should I wear today or were can i go today?"**
> ClimaPlan auto-detects their location, pulls a decade of weather history, spots anomalies, and delivers a plain-language answer — with safety tips and outfit suggestions — in one command.

---

## What It Does

ClimaPlan is an AI weather intelligence agent built specifically for agricultural communities. You run it, and it:

1. **Finds where you are** — no input needed, it figures out your location automatically
2. **Gets today's weather** — live temperature, wind speed, and conditions
3. **Digs into 10 years of history** — downloads a decade of daily weather data for your exact coordinates
4. **Spots what's unusual** — generates Python code on the fly to flag when today's readings are statistically out of the ordinary (e.g. "temperature is 2 standard deviations above the historical average for this week")
5. **Makes recommendations** — what to wear, where to go, what to avoid
6. **Reflects and improves** — the AI critiques its own recommendations, then asks if you want the upgraded version
7. **Writes a report** — delivers one clear, friendly summary a non-technical person can act on

---

## Why It Matters — The Problem for Nigerian Farmers

Nigerian smallholder farmers lose between **$3–9 billion annually** to weather-related crop failures. The core issue is not that weather data doesn't exist — it does. The problem is that:

- Weather forecasts are written for urban audiences, not farm decisions
- Historical climate baselines are locked inside academic databases
- Farmers can't afford agronomists to interpret data for them
- Anomalous weather events (late rains, unexpected dry spells) aren't flagged until it's too late

ClimaPlan collapses the gap between raw climate data and a usable "plant now or wait" decision. It runs on a laptop with a single command, requires no weather expertise, and speaks in plain language.

---

## Architecture — Full Pipeline

```
┌─────────────────────────────────────────────────────────────────────┐
│                        MAIN GRAPH                                   │
│                                                                     │
│  START                                                              │
│    │                                                                │
│    ▼                                                                │
│  [get_location]  ──── ipinfo.io API ──── Guardrails validation      │
│    │                  (lat/lon/city)     (Pydantic schema check)    │
│    │                                                                │
│    ▼                                                                │
│  [get_weather]   ──── Open-Meteo API ── Today's temp/wind/code     │
│    │                                                                │
│    ▼                                                                │
│  [weather_history] ── Open-Meteo Archive ── 10 years of daily data │
│    │                   (temp_max, temp_min, precip,                │
│    │                    wind_speed, shortwave_radiation)            │
│    ▼                                                                │
│  [format_response] ── LLM ── Human-readable weather summary        │
│    │                                                                │
│    ▼                                                                │
│  [trim_message]  ──── LLM ── Compress message history              │
│    │                                                                │
│    ▼                                                                │
│  [persist_data]  ───────────────────────────────────────────────┐  │
│    │              calls SUBGRAPH 1 + SUBGRAPH 2                  │  │
│    │                                                             │  │
│    │   ┌─────────────────────────────────────────────────────┐  │  │
│    │   │  SUBGRAPH 1: Save & Analyse                         │  │  │
│    │   │   [save_file] → writes historical data to .csv      │  │  │
│    │   │       │                                             │  │  │
│    │   │       ▼                                             │  │  │
│    │   │   [analyse_data] → calls SUBGRAPH 2                 │  │  │
│    │   └─────────────────────────────────────────────────────┘  │  │
│    │                                                             │  │
│    │   ┌─────────────────────────────────────────────────────┐  │  │
│    │   │  SUBGRAPH 2: CodeAct Analysis (LLM writes + runs    │  │  │
│    │   │              Python to find anomalies)              │  │  │
│    │   │                                                     │  │  │
│    │   │   [create_code] ── LLM generates analysis Python    │  │  │
│    │   │       │                                             │  │  │
│    │   │       ▼                                             │  │  │
│    │   │   [execute_code] ── runs code in sandbox            │  │  │
│    │   │       │                                             │  │  │
│    │   │       ▼                                             │  │  │
│    │   │   [process_results] ── validates RESULT dict        │  │  │
│    │   │       │                                             │  │  │
│    │   │    (error?)──►[adjust_code] ──► retry (max 3x)      │  │  │
│    │   │       │                                             │  │  │
│    │   │    (3 fails)─►[compute_baseline] ── LLM fallback    │  │  │
│    │   │       │                                             │  │  │
│    │   │      END  (analysis + anomaly + stats)              │  │  │
│    │   └─────────────────────────────────────────────────────┘  │  │
│    └────────────────────────────────────────────────────────────┘  │
│    │                                                                │
│    ▼                                                                │
│  [recommend]  ──────────────────────────────────────────────────┐  │
│    │           calls SUBGRAPH 3                                  │  │
│    │   ┌─────────────────────────────────────────────────────┐  │  │
│    │   │  SUBGRAPH 3: Recommend + Reflect                    │  │  │
│    │   │   [recommend] ── LLM → structured Recommendation    │  │  │
│    │   │       │                                             │  │  │
│    │   │       ▼                                             │  │  │
│    │   │   [reflect] ─── LLM self-critiques recommendations  │  │  │
│    │   │       │                                             │  │  │
│    │   │       ▼                                             │  │  │
│    │   │   [ask_user] ── "Apply improved version? (Y/N)"     │  │  │
│    │   └─────────────────────────────────────────────────────┘  │  │
│    └────────────────────────────────────────────────────────────┘  │
│    │                                                                │
│    ▼                                                                │
│  [final_output] ── LLM ── Combines everything into one report      │
│    │                                                                │
│   END                                                               │
└─────────────────────────────────────────────────────────────────────┘

  ── Every node feeds into error_handler on failure ──► END
```

**MCP Tool Layer** (all nodes talk to tools via the Model Context Protocol):

| MCP Server | Transport | Purpose |
|---|---|---|
| `climate_report_srv` | stdio | IP-based location lookup, today's weather |
| `open_meteo` | stdio | 10-year historical weather archive |
| `weather` | stdio | Live current conditions |

---

## Tech Stack

| Layer | Technology | Role |
|---|---|---|
| **LLM** | Groq `llama-3.1-8b-instant` (default) / Google Gemini | Reasoning, code generation, text output |
| **Orchestration** | LangGraph + LangChain | Multi-graph state machine with conditional routing |
| **MCP Tools** | FastMCP, `mcp-weather-server`, custom Open-Meteo adapter | Expose weather APIs as callable tools |
| **Structured Output** | Pydantic v2 models | Schema enforcement on every LLM output |
| **Input Validation** | Guardrails AI | Validates location data before it enters the graph |
| **CodeAct Execution** | `exec()` in restricted sandbox | Runs LLM-generated Python safely (pandas/numpy only) |
| **Embeddings / Vector Store** | — | Not used; anomaly detection is code-generated statistics |
| **Data Processing** | Pandas, NumPy | DataFrame operations for z-score / percentile analysis |
| **Location** | ipinfo.io API | Automatic IP-to-coordinates resolution |
| **Weather Data** | Open-Meteo API (free, no key) | Archive (10yr) + forecast (live) endpoints |
| **Observability** | LangSmith | Full trace of every graph run; LLM-as-judge eval suite |

---

## What I Learned / Challenges

### 1. CodeAct is powerful but brittle on small models
The LLM generates Python to analyse a CSV — z-scores, percentiles, anomaly flags — and then calls `execute_code()` with that code as a string argument. This works well with larger models, but `llama-3.1-8b-instant` on Groq would occasionally output `<function=execute_code>...</function>` as raw text instead of a proper tool call. The fix was a three-layer resilience system: up to 3 retry attempts where the error message is fed back to the LLM to self-correct, followed by a direct structured-output fallback that asks the LLM to compute the baseline without code at all. The lesson: CodeAct requires either a capable model or a strong recovery path.

### 2. Sandboxing `exec()` without breaking pandas
To run LLM-generated code safely, I restricted `__builtins__` to only `__import__`, `print`, and `range`, then explicitly injected `pd`, `np`, and `datetime` into the execution environment. The problem: pandas internally calls stdlib functions that aren't in the restricted namespace (e.g. `len`, `isinstance`, `type`). The solution was to pass the full builtins via `__import__` and only block dangerous calls by design rather than by stripping builtins entirely. Getting this balance right took multiple iterations.

### 3. Async MCP + sync LangGraph subgraphs don't mix cleanly
The main graph is `async` (awaits MCP tool calls over stdio). The analysis subgraphs (`save_file_subgraph`, `analysis_subgraph`) are invoked synchronously from within an async node using `.invoke()`. This caused event loop conflicts. The solution was to keep subgraph invocation inside a sync wrapper function that runs within the async node's thread, effectively isolating the sync/async boundary. It works but is fragile — the right long-term fix is to make all subgraphs async.

### 4. Guardrails validation as a circuit-breaker, not just schema checking
I initially used Pydantic validation directly. Swapping to Guardrails AI for the location schema unlocked two things: detailed `guard_notes` explaining *why* validation failed (useful for debugging malformed ipinfo responses), and a clean integration point to add custom validators (e.g. blocking coordinates that fall in the ocean) without touching the graph logic. The discipline of validating at the boundary — before coordinates enter the graph — prevented an entire class of silent downstream failures.

### 5. The reflection loop needs a stopping condition you define explicitly
The `recommend → reflect → ask_user` subgraph works because the LLM is asked to critique its own output against the weather data, not just rephrase it. However, without careful prompting, the reflection step would "improve" recommendations into hallucinated specifics (e.g. inventing humidity values that weren't in the input). The fix was to constrain the reflection prompt to only use facts *implied by the weather data that was provided*, and to make the `ask_user` node the human-in-the-loop gate before any reflected recommendations enter the final output. Human oversight as a guardrail is simpler than prompt engineering alone.

---

## Demo

> Coming soon — deployment link will appear here once hosted.

---

## How to Run

### Prerequisites

- Python 3.13+
- [`uv`](https://github.com/astral-sh/uv) package manager
- A free [Groq API key](https://console.groq.com) (or a Gemini API key)
- Optional: [LangSmith](https://smith.langchain.com) account for tracing

### 1. Clone the repo

```bash
git clone https://github.com/your-username/ClimaPlan.git
cd ClimaPlan
```

### 2. Install dependencies

```bash
uv sync
```

### 3. Set up environment variables

Create a `.env` file in the project root:

```env
# Required: choose one LLM provider
GROQ_API_KEY=your_groq_key_here

# Optional: switch to Gemini instead (update config.py to use ChatGoogleGenerativeAI)
GEMINI_API_KEY=your_gemini_key_here
GEMINI_API_MODEL=gemini-2.0-flash

# Optional: more accurate location detection
IPINFO_TOKEN=your_ipinfo_token_here

# Optional: enable LangSmith tracing
LANGCHAIN_TRACING_V2=true
LANGCHAIN_API_KEY=your_langsmith_key_here
LANGCHAIN_PROJECT=climaplan
```

### 4. Run the agent

```bash
uv run python main.py
```

The agent will auto-detect your location, fetch weather data, run the full analysis pipeline, and print a weather report to the terminal. When the reflection step completes, it will ask you:

```
Do you want to apply the improved recommendations? (Yes/No):
```

### 5. (Optional) Run the evaluation suite

```bash
uv run python eval.py
```

This creates a LangSmith dataset, runs the full graph, and scores code quality + reflection quality using LLM-as-judge evaluators. Results are visible in your LangSmith dashboard.

---

## Project Structure

```
ClimaPlan/
├── main.py              # Main graph + all 4 subgraphs
├── index.py             # Alternative ReAct agent variant of main graph
├── config.py            # LLM and MCP client configuration
├── models.py            # Pydantic state schemas for all graphs
├── prompts.py           # All LLM prompt templates
├── tools.py             # FastMCP server (location + weather tools)
├── filter_open_meteo.py # Open-Meteo archive MCP adapter
├── utility.py           # Parsing helpers for MCP responses
├── baseline.py          # Reference Python code for CodeAct evaluation
├── eval.py              # LangSmith evaluation suite
├── data/                # Auto-created: historical weather CSVs
├── weather_graph.png    # Auto-generated graph diagram (main)
├── analysis_graph.png   # Auto-generated graph diagram (CodeAct)
├── recommend_graph.png  # Auto-generated graph diagram (reflection)
└── save_file_graph.png  # Auto-generated graph diagram (persistence)
```

Built by Aghogho Joy Olokpa — connect with me on LinkedIn https://www.linkedin.com/in/aghogho-olokpa-1b0b11115.