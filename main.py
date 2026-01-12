# import MessagesState
from langgraph.graph import StateGraph, START, END
from langgraph.graph import MessagesState
import os
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.prebuilt import ToolNode, InjectedState
from langchain.messages import HumanMessage, SystemMessage, AnyMessage
from pydantic import BaseModel
from typing import Optional, Dict, Any, List, Annotated
from langchain.messages import ToolMessage
from utility import _tool_content_to_dict, _get_trailing_tool_messages, inc_count
from langchain_groq import ChatGroq
from langchain_core.runnables import RunnableConfig
from langgraph.graph.message import add_messages 
import json
import sys as system


import asyncio
load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")
api_key2= os.getenv("GEMINI_API_KEY2")
api_base = os.getenv("GEMINI_API_BASE")
model = os.getenv("GEMINI_API_MODEL")
groq_api_key = os.getenv("GROQ_API_KEY")

# Validate required environment variables
if not all([api_key, model, api_key2]):
    raise ValueError("Missing required environment variables: GEMINI_API_KEY, GEMINI_API_MODEL")

# Initialize the Google Generative AI LLM
# llm = ChatGoogleGenerativeAI(
#     model=model,
#     google_api_key=api_key2,
#     temperature=0
# )

# Alternatively, initialize the Groq LLM
llm = ChatGroq(
    model="llama-3.1-8b-instant",
    groq_api_key=os.getenv("GROQ_API_KEY"),
    temperature=0.2
)

# Initialize the MCP client
mcp_client  = MultiServerMCPClient(
    {
        "climate_report_srv": {
            "transport": "stdio",
            "command": "uv",
            "args": ["run", "python", "tools.py"],
        },
         "open_meteo": {
            "transport": "stdio",
            "command": "npx",
            "args": ["open-meteo-mcp-server"],
            "env": {
                "OPEN_METEO_ARCHIVE_API_URL": "https://archive-api.open-meteo.com/v1/archive",
            }
        }

    }
)

class AppState(BaseModel):
    retries: int = 0
    user_date: Optional[str] = None
    user_location_text: Optional[str] = None
    latitude: Optional[float] = None
    longitude: Optional[float] = None

    today_weather: Optional[Dict[str, Any]] = None
    hist_weather: Optional[List[Dict[str, Any]]] = None

    analysis: Optional[Dict[str, Any]] = None
    draft: Optional[str] = None
    reflection_notes: Optional[str] = None
    approved: bool = False



class WeatherState(BaseModel):
    retries: int = 0
    messages: Annotated[ list[AnyMessage], add_messages] = []
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    city: str | None = None
    country: str | None = None
    today_weather: Optional[Dict[str, Any]] = None
    user_date: Optional[str] = None
    hist_weather: Optional[Any] = None  # keep flexible (list/dict) depending on your MCP tool
    ready_to_format: bool = False 

    # final strict output (as dict)
    final: Optional[Dict[str, Any]] = None


# System prompt to guide the LLM to call tools in sequence
system_prompt = """You are a helpful weather assistant. When a user asks about weather, you should:

1. First, call the `get_user_location` tool to get the user's current location (latitude and longitude)
2. Then, use the latitude and longitude from the location result to call `get_weather_for_today` tool. 
3. Finally, provide a friendly response about the weather based on the data you received

Always call get_user_location first, then get_weather_for_today with the coordinates from the location data."""




# -----------------------------
# ROUTERS
# -----------------------------
# Routing function after agent node to ensure the correct flow of the graph
def route_after_agent(state: WeatherState) -> str:
    """If agent asked for tools -> tools; else -> reflection."""
    last = state.messages[-1]
    tool_calls = getattr(last, "tool_calls", None) or []
    return "tools" if tool_calls else "reflection"
# Routing function after reflection node to decide next step
def route_after_reflection(state: WeatherState) -> str:
    if state.ready_to_format:
        return "format_response"

    if state.retries >= 3:
        return "error_handler"

    return "retry"

# Reflection node to check if required data is present before formatting to format_response node
def reflection_node(state: WeatherState) -> dict:
    """
    Reflection = deterministic check that required data exists.
    For some runs it is not going to get_weather_today which is required if not it fails at formatting.
    """
    # print("Running reflection node...")
    # print("Current state:", state)
    missing = []
    if state.latitude is None or state.longitude is None:
        missing.append("location coordinates is missing")
    if state.today_weather is None:
        missing.append("today weather is missing")

    ready = len(missing) == 0
    return {
        "ready_to_format": ready,
        "reflection_notes": "OK" if ready else f"Missing: {', '.join(missing)}"
    }



# -----------------------------
# Main Graph Creation
# -----------------------------
# Create the weather retrieval graph
async def create_graph():
    """Create the graph after getting tools"""
    alltools = await mcp_client.get_tools()
    # Filter tools to only include those we need to keep the graph efficient and not overloaded
    ALLOW = {"weather_archive", "geocoding", "get_user_location", "get_weather_for_today"}  # only what you need
    tools= [t for t in alltools if t.name in ALLOW]

    # Debug: Print tools to verify they're loaded
    print(f"Loaded {len(tools)} tools:")
    for tool in tools:
        print(f"  - {tool.name}: {tool.description}")

    async def get_weather(state: WeatherState) -> dict:
        # Build messages list - include system prompt and all previous messages
        messages = list(state.messages)
        
        # Add system message if not already present
        if not any(isinstance(msg, SystemMessage) for msg in messages):
            messages = [SystemMessage(content=system_prompt)] + messages
        
        # Invoke LLM with tools and all context
        result = await llm.bind_tools(tools).ainvoke(messages)
        return {"messages": [result]}
    
    # Error handler node to provide a fallback message
    async def error_handler(state: WeatherState) -> dict:
        msg = HumanMessage(
            content="I couldn’t fetch live weather data right now. Please try again shortly."
        )
        return {"messages": [msg]}
    
    # Condition to exit loop after a certain number of iterations the graph seems to be stuck in calling the agent
    def loop_counter_condition(state: WeatherState) :
        retries = state.retries + 1
        print("In loop counter condition. Count is:", retries)
        return {"retries": retries}

    # Format response node to create a human-readable weather summary
    async def format_response(state: WeatherState) -> dict:
        """Format the weather data into human-readable format"""
        weather_data = state.today_weather
        city = state.city
        country = state.country

        formatting_prompt = HumanMessage(
            content=(
                "Write a friendly, human-readable weather summary for a regular person. "
                "Describe how it feels (cold/warm), whether it's windy, rainy, or sunny. "
                "Use the location if available.\n\n"
                f"Location: {city}, {country}\n"
                f"Weather data: {weather_data}"
            )
        )

        # Create a prompt that includes the weather data and asks for formatting
        formatted_result = await llm.ainvoke([formatting_prompt])
        return {"messages": [formatted_result]}

    
    async def update_weather_state(state: WeatherState) -> dict:
        """
        After ToolNode runs, update WeatherState fields for each ToolMessage appended.
        """
        messages = state.messages
        tool_msgs = _get_trailing_tool_messages(messages)
        if not tool_msgs:
            return {}   

        updates: Dict[str, Any] = {}

        for tm in tool_msgs:
            tool_name = tm.name
            payload = _tool_content_to_dict(tm.content)
            print(f"Processing tool '{tool_name}' with payload: {payload}")

                # Extract JSON string from tool result
            text_block = payload["value"][0]["text"]
                # Parse JSON inside "text"
            parsed = json.loads(text_block)

            # ---- Map MCP tool outputs to WeatherState fields ----
            if tool_name == "get_user_location":

                updates["latitude"] = float(parsed.get("latitude"))
                updates["longitude"] = float(parsed.get("longitude"))

                updates["city"] = parsed.get("city")
                updates["country"] = parsed.get("country")

            elif tool_name == "get_weather_for_today":
                updates["today_weather"] = parsed

        return updates

    weather_graph = StateGraph(WeatherState)
    weather_graph.add_node("agent",get_weather)
    weather_graph.add_node("tools", ToolNode(tools))
    weather_graph.add_node("update_state", update_weather_state)
    weather_graph.add_node("reflection", reflection_node)
    weather_graph.add_node("retry", loop_counter_condition)
    weather_graph.add_node("error_handler", error_handler)
    weather_graph.add_node("format_response", format_response)

    weather_graph.add_edge(START, "agent")
    weather_graph.add_conditional_edges(
        "agent",
        route_after_agent,
        path_map={"tools": "tools", "reflection": "reflection"}
    )
    weather_graph.add_edge("tools", "update_state")
    weather_graph.add_edge("update_state", "reflection")
    weather_graph.add_conditional_edges(
        "reflection",
        route_after_reflection,
        path_map={
            "format_response": "format_response",
            "retry": "retry",
            "error_handler": "error_handler",
        },
    )
    weather_graph.add_edge("retry", "agent")

    weather_graph.add_edge("format_response", END)
    weather_graph.add_edge("error_handler", END)

    return weather_graph.compile()


async def main():
    compiled_graph = await create_graph()
    # Example usage - you'll want to modify this based on your needs
    # Or save as PNG (requires pygraphviz)
    try:
        compiled_graph.get_graph().draw_mermaid_png(output_file_path="weather_graph.png")
        print("\nGraph saved as weather_graph.png")
    except Exception as e:
        print(f"\nCould not save PNG (pygraphviz may not be installed): {e}")
    result = await compiled_graph.ainvoke({
        "messages": [HumanMessage(content="What's the weather today?")]
    })
    
    # print(result)
    #print(result["messages"][-5:]) 

if __name__ == "__main__":
    asyncio.run(main())

