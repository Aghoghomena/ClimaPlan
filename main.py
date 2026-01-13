# import MessagesState
from click import prompt
from langgraph.graph import StateGraph, START, END
from langgraph.graph import MessagesState
import os
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.prebuilt import ToolNode, InjectedState
from langchain.messages import HumanMessage, SystemMessage, AnyMessage
from pydantic import BaseModel,Field, ValidationError, field_validator
from typing import Optional, Dict, Any, List, Annotated
from langchain.messages import ToolMessage
from utility import _tool_content_to_dict,parse_mcp_text_result , weather_data_to_df,_parse_payload, parse_mcp,get_baseline_code
from langchain_groq import ChatGroq
from langchain_core.runnables import RunnableConfig
from langgraph.graph.message import add_messages 
import json
import sys as system
from datetime import datetime, timedelta
from guardrails import Guard
from guardrails.errors import ValidationError as GuardrailsValidationError
from guardrails.types import OnFailAction
import re
import pandas as pd
import numpy as np
from langchain_core.tools import tool
from langsmith import traceable
import random
import builtins
from langchain_core.runnables import RunnableConfig
from langgraph.types import interrupt

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
#     google_api_key=api_key,
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
            "command": "python",
            "args": ["filter_open_meteo.py"],
             "env": {
                "OPEN_METEO_ARCHIVE_API_URL": "https://archive-api.open-meteo.com"
             }
        },
        "weather": {
            "transport": "stdio",
            "command": "python",
            "args": ["-m", "mcp_weather_server"],
        }
        

    }
)




class WeatherState(BaseModel):
    retries: int = 0
    messages: Annotated[ list[AnyMessage], add_messages] = []
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    city: str | None = None
    country: str | None = None
    today_weather: Optional[Dict[str, Any]] = None
    formated_today_weather: str | None = None
    hist_weather: Optional[Any] = None  # keep flexible (list/dict) depending on your MCP tool
    ready_to_format: bool = False 
    Error: str | None = None
    generated_code: str | None = None
    analysis: str | None = None
    anomaly: Optional [Dict[str, Any]] = None
    stats: Optional [Dict[str, Any]] = None
    recommendations: Optional[List[str]] = None
    final_output: str | None = None

    

    # final strict output (as dict)
    final: Optional[Dict[str, Any]] = None

class UserLocationSchema(BaseModel):
    latitude: float = Field(..., ge=-90, le=90)
    longitude: float = Field(..., ge=-180, le=180)
    city: str | None = None
    country: str | None = None

class WeatherAnalysisOutPut(BaseModel):
    analysis: str
    anomaly: Dict[str, Any]          # e.g. {"temp_anomaly_c": 1.2, "precip_anomaly_mm": -0.3, ...}
    stats: Dict[str, Any] 

class WeatherAnalysis(BaseModel):
    analysis: str | None = None
    anomaly: Dict[str, Any]  = None       # e.g. {"temp_anomaly_c": 1.2, "precip_anomaly_mm": -0.3, ...}
    stats: Dict[str, Any] = None
    hist_weather: Optional[Any] = None 
    generated_code: str | None = None
    messages: Annotated[list[AnyMessage], add_messages] = []
    hist_file_path: Optional[str] = None
    execution_retries: int = 0  # Track retries for code execution
    max_execution_retries: int = 3  # Max retries for execution
    last_error: Optional[str] = None  # Store last error message

class Recommendation(BaseModel):
    recommendations: List[str] = Field(description="General advice based on the weather where to go what to eat, what to wear.")
    safety_notes: List[str] = Field(description="Safety/health/travel notes if any.")
    recommendation_summary: List[str] = Field(description="Summary of reccomendation, outfit_suggestions, safety_notes.")
    Error: str | None = None

class Reflections(BaseModel):
    recommendations: List[str] = Field(description="General advice based on the weather where to go what to eat, what to wear.")
    safety_notes: List[str] = Field(description="Safety/health/travel notes if any.")
    recommendation_summary: List[str] = Field(description="Summary of reccomendation, outfit_suggestions, safety_notes.")
    Error: str | None = None
    reflection_notes: str = Field(description="Notes about what was reflected/improved") 
    improvements: str = Field(description="Specific suggestions to improve the recommender")

# Add state model for recommendation subgraph (around line 147, after Reflections)
class RecommendationState(BaseModel):
    today_weather: Optional[Dict[str, Any]] = None
    formated_today_weather: Optional[str] = None
    original_recommendations: Optional[Recommendation] = None
    reflected_recommendations: Optional[Reflections] = None
    final_recommendations: Optional[List[str]] = None
    user_decision: Optional[str] = None  # "apply" or "keep_original"
    reflection_notes: Optional[str] = None
    messages: Annotated[list[AnyMessage], add_messages] = []
    Error: Optional[str] = None


# Configuration for file saving
STORAGE_DIR = "data"  # Directory to save files
FILE_FORMAT = "csv"  # Default format: csv, parquet, or json

# System prompt to guide the LLM to call tools in sequence
system_prompt = """You are a helpful weather assistant. When a user asks about weather, you should:

1. First, call the `get_user_location` tool to get the user's current location (latitude and longitude)
2. Then, use the latitude and longitude from the location result to call `get_weather_for_today` tool. 
3. Finally, provide a friendly response about the weather based on the data you received

Always call get_user_location first, then get_weather_for_today with the coordinates from the location data."""




# -----------------------------
# ROUTERS
# -----------------------------
# Routing function to go to errror and end if any of the tools or nodes fails
def check_for_errors(state: WeatherState) -> str:
    """Check if any node set an error and route to error handler"""
    if state.Error:
        return "error_handler"
    return "continue"  # Continue normal flow


#routing for subgraph 2
def route_after_execution(state: WeatherAnalysis) -> str:
    """Route after code execution - check for errors and retries"""
    # Check if there's an error and we can retry
    if state.last_error and state.execution_retries < state.max_execution_retries:
        return "adjust_code"  # Retry with adjusted code
    
    # Check if there's a final error (max retries exceeded) - go to direct computation
    if state.last_error and state.execution_retries >= state.max_execution_retries:
        return "compute_baseline"  
    
    # Success - no errors
    return END


# -----------------------------
# GuardRails
# -----------------------------
location_guard = Guard.for_pydantic(UserLocationSchema)
    




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
    # print(f"Loaded {len(tools)} tools:")
    # for tool in tools:
    #     print(f"  - {tool.name}: {tool.description}")

    tool_map = {tool.name: tool for tool in tools}

    get_user_location_tool = tool_map["get_user_location"]
    get_weather_for_today_tool = tool_map["get_weather_for_today"]
    weather_archive_tool = tool_map["weather_archive"]

    async def user_location(state: WeatherState) -> dict:

        try:
            result = await get_user_location_tool.ainvoke({})
            raw_data = result[0]["text"] if isinstance(result, list) else result
            # Validate at this point
            outcome = location_guard.parse(raw_data)
            data = outcome.validated_output
            return {
                "latitude": data["latitude"],
                "longitude": data["longitude"],
                "city": data["city"],
                "country": data["country"],
                "Error": None
            }
        except (json.JSONDecodeError, ValidationError, ValueError) as e:
        # structured error you can route on
            return {
                "Error": f"user_location validation failed: {str(e)}",
                "guard_ok": False,
                "guard_notes": [str(e)],
            }
        
        except Exception as e:
            print(f"Error in user_location: {e}")
            import traceback
            traceback.print_exc()
            return {"Error":  f"Error in user_location: {e}"}

    
    async def get_weather_for_today(state: WeatherState) -> dict:
        try:
            raw = await get_weather_for_today_tool.ainvoke({
                "latitude": state.latitude,
                "longitude": state.longitude
            })
            data = raw[0] if isinstance(raw, list) else raw

            return {
                "today_weather": data,
                "Error": None  # Clear any previous error
            }
        except Exception as e:
            print(f"Error in get_weather_for_today: {e}")
            import traceback
            traceback.print_exc()
            return {"Error":  f"Error in get_weather_for_today: {e}"}

    

    async def get_historical_data(state: WeatherState) -> dict:
        
        try:

            # Get latitude and longitude from state
            user_latitude = state.latitude
            user_longitude = state.longitude

            if user_latitude is None or user_longitude is None:
                # If we don't have location yet, return empty (shouldn't happen in normal flow)
                return {"Error": "Missing latitude/longitude"}
            
            # Calculate dates (last 7 days)
            #end_date = "2025-12-31" #using this date because they dont have the correct date yet
            end_date = datetime.now().strftime('%Y-%m-%d')
            start_date = (datetime.now() - timedelta(days=365 * 10)).strftime('%Y-%m-%d')
            # Call the tool programmatically
            tool_out = await weather_archive_tool.ainvoke({
                    "latitude": user_latitude,
                    "longitude": user_longitude,
                    "daily": [
                        "temperature_2m_max",
                        "temperature_2m_min",
                        "precipitation_sum",
                        "wind_speed_10m_max",
                        "shortwave_radiation_sum"
                        ],
                    "start_date":start_date,
                    "end_date": end_date,
                    "temperature_unit": "celsius",
                    "timezone": "Europe/Dublin",
                    
            })
            # tool_out might be list[{"type":"text","text":"..."}] or a ToolMessage.

            data = parse_mcp(tool_out)               # dict
            df = weather_data_to_df(data)           # your helper
            records = df.to_dict(orient="records")  # serialize-friendlye

            return {"hist_weather": df, "Error": None}

        except Exception as e:
            print(f"Error calling weather_archive: {e}")
            return {"Error": f"Error calling weather_archive: {e}"}


    
    # Error handler node to provide a fallback message
    async def error_handler(state: WeatherState) -> dict:
        msg = HumanMessage(
            content="I couldn’t fetch live weather data right now. Please try again shortly."
        )
        return {"messages": [msg]}


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
         # Extract the content from the LLM response
        formatted_content = formatted_result.content if hasattr(formatted_result, 'content') else str(formatted_result)
    
        return {"messages": [formatted_result],"formated_today_weather": formatted_content  }
    
    async def trim_message(state: WeatherState) -> dict:
         # Convert messages to dict format (JSON serializable)
        messages_content = []
        for msg in state.messages:
            if hasattr(msg, 'content'):
                messages_content.append({
                    "type": msg.__class__.__name__,
                    "content": msg.content
                })
            else:
                messages_content.append({"type": msg.__class__.__name__, "content": str(msg)})
        
        msg = HumanMessage(content=f"Summarize :\n{json.dumps(messages_content, indent=2)}")
        result = await llm.ainvoke([msg])
        return {"messages": [result]}
    
    # get analysis Data save the data to a csv and call the codeAct Agent
    def persist_graph(state: WeatherState) -> dict:

        result = save_file_subgraph.invoke(WeatherAnalysis(hist_weather=state.hist_weather))
        # Extract results from analysis_subgraph output
        # result will be a WeatherAnalysis object/dict with analysis, anomaly, stats, etc.
        if isinstance(result, dict):
            analysis_result = result.get("analysis", {})
            if isinstance(analysis_result, dict):
                # Extract the analysis results
                return {
                    "analysis": analysis_result.get("analysis"),
                    "anomaly": analysis_result.get("anomaly"),
                    "stats": analysis_result.get("stats"),
                    "generated_code": analysis_result.get("generated_code"),
                    "Error": analysis_result.get("Error")
                }
            else:
                # If analysis is not a dict, use the result directly
                return {
                    "analysis": analysis_result,
                    "anomaly": result.get("anomaly"),
                    "stats": result.get("stats"),
                    "generated_code": result.get("generated_code"),
                    "Error": result.get("Error")
                }
        else:
            # Fallback if result format is unexpected
            return {
                "analysis": str(result) if result else None,
                "Error": "Unexpected result format from analysis subgraph"
            }
        
    async def call_recommendation(state: WeatherState)-> dict:
            try:
                # Pass only weather data
                result = recommend_graph.invoke({
                    "today_weather": state.today_weather,
                    "formated_today_weather": state.formated_today_weather
                })
                
                # Extract recommendations from the subgraph result
                # Since recommend_for_weather already updated the state, extract it
                final_recommendations = result.get("final_recommendations", [])
                
                return {
                    "recommendations": final_recommendations,
                    "Error": result.get("Error")
                }
            except Exception as e:
                return {
                    "recommendations": [],
                    "Error": f"Error in call_recommendation: {e}"
                }
            
    #final format to give an output for the user
    async def final_output(state: WeatherState) -> str:
        try:
            # Extract data from state
            recommendations = state.recommendations or []
            analysis = state.analysis or "No analysis available"
            anomaly = state.anomaly or {}
            city = state.city or "your location"
            country = state.country or ""
            location = f"{city}, {country}" if country else city
            # Format recommendations
            recs_text = "\n".join([f"- {rec}" for rec in recommendations]) if recommendations else "No recommendations available"
            
            # Format anomaly data
            anomaly_text = ""
            if anomaly:
                anomaly_items = [f"{key}: {value}" for key, value in anomaly.items()]
                anomaly_text = "\n".join(anomaly_items)
            else:
                anomaly_text = "No significant anomalies detected"

            # Create prompt for final output
            final_prompt = f"""Create a comprehensive, cohesive weather report that combines all the information below.
                    Location: {location}
                    Weather Analysis:{analysis}
                    Statistical Summary:{json.dumps(analysis, indent=2) if analysis else "No statistics available"}
                    Anomalies Detected:{anomaly_text}
                    Recommendations:{recs_text}

                    Generate a well-structured, easy-to-read final report that:
                    1. Summarizes the current weather situation
                    2. Explains the analysis findings
                    3. Highlights any anomalies or unusual patterns
                    4. Provides clear, actionable recommendations
                    5. Is friendly and accessible to a general audience

                    Keep it concise but comprehensive."""
            # Generate final output using LLM
            final_result = await llm.ainvoke([HumanMessage(content=final_prompt)])
            final_content = final_result.content if hasattr(final_result, 'content') else str(final_result)
            return{"final_output": final_content,
                   "messages": [HumanMessage(content=final_content)]}
        except Exception as e:
            error_msg = f"Error generating final output: {e}"
            return {
                "Error": error_msg,
                "final": {
                    "error": error_msg,
                    "recommendations": state.recommendations or [],
                    "analysis": state.analysis,
                    "anomalies": state.anomaly
                }
        }
            

    

    weather_graph = StateGraph(WeatherState)
    weather_graph.add_node("get_location", user_location)
    weather_graph.add_node("get_weather", get_weather_for_today)
    weather_graph.add_node("weather_history", get_historical_data)
    weather_graph.add_node("format_response", format_response)
    weather_graph.add_node("error_handler", error_handler)
    weather_graph.add_node("trim_message", trim_message)
    weather_graph.add_node("persist_data", persist_graph)
    weather_graph.add_node("recommend", call_recommendation)
    weather_graph.add_node("final_output", final_output)  

    weather_graph.add_edge(START, "get_location")
    # After each node, check for errors
    weather_graph.add_conditional_edges(
        "get_location",
        check_for_errors,
        path_map={"error_handler": "error_handler", "continue": "get_weather"}
    )
    weather_graph.add_conditional_edges(
    "get_weather",
    check_for_errors,
    path_map={"error_handler": "error_handler", "continue": "weather_history"}
    )
    weather_graph.add_conditional_edges(
        "weather_history",
        check_for_errors,
        path_map={"error_handler": "error_handler", "continue": "format_response"}
    )
    weather_graph.add_edge("format_response", "trim_message")
    weather_graph.add_edge("trim_message", "persist_data")
    weather_graph.add_edge("persist_data", "recommend")
    weather_graph.add_edge("recommend", "final_output") 
    weather_graph.add_edge("final_output", END) 
    weather_graph.add_edge("error_handler", END)

    return weather_graph.compile()



# -----------------------------
# SUBGRAPH 2
# -----------------------------
# Save historical data into a file

def save_file(state: WeatherAnalysis, str = FILE_FORMAT, storage_dir: str = STORAGE_DIR) -> Dict[str, Any]:
        try:
            df = state.hist_weather

             # Create storage directory if it doesn't exist
            os.makedirs(storage_dir, exist_ok=True)

            #generate randon uuid
            randomnum = random.randint(1, 100000)
            file_name = f"{randomnum}"

            filename = f"historical_weather_compare_{file_name}"
            path = os.path.join(storage_dir, f"{filename}.csv")
            df.to_csv(path, index=False)

            return {"hist_file_path": path, "Error": None}
        except Exception as e:
            return {"Error": f"persist_subgraph failed: {e}"}
        
# get analysis Data
def analyse_data(state: WeatherAnalysis) -> dict:

        result = analysis_subgraph.invoke(state)
        # Return the full result so persist_graph can extract what it needs
        # This will be merged into the state by save_file_subgraph
        return {
            "analysis": result.get("analysis"),
            "anomaly": result.get("anomaly"),
            "stats": result.get("stats"),
            "generated_code": result.get("generated_code"),
            "Error": result.get("Error")
        }


subgraph1 = StateGraph(WeatherAnalysis)
subgraph1.add_node("save_file",save_file )
subgraph1.add_node("analyse_data", analyse_data)

subgraph1.add_edge(START, "save_file")
subgraph1.add_edge("save_file", "analyse_data")
save_file_subgraph = subgraph1.compile()

# Or save as PNG (requires pygraphviz)
try:
    save_file_subgraph.get_graph().draw_mermaid_png(output_file_path="save_file_graph.png")
    print("\nGraph saved as save_file_graph.png")
except Exception as e:
    print(f"\nCould not save PNG (pygraphviz may not be installed): {e}")

# -----------------------------
# SUBGRAPH 2
# -----------------------------


# -----------------------------
# SUBGRAPH 3
# -----------------------------
#CodeACt to compute and do the analysis


CODEACT_PROMPT = """
You are an expert Python programmer and data scientist.

Your task:
1) Understand the data at the file path
2) generate python code to cmpute anomaly, median, mean and percentile of historical data against todays weather
   - Reads the CSV file from the file_path provided
   - Identify the row with the date today using:
     The column that has date of today
     Else treat the MAX(date/time) as today (latest timestamp)
   - For each numeric weather metric column (all numeric columns except obvious IDs):
     * Compute baseline statistics over historical rows: mean, median, std (sample std), min, max
     * Compute z_score = (today - mean) / std if std > 0 else null
     * Compute anomaly flag:
       - "high" if z_score >= +2
       - "low" if z_score <= -2
       - else null
3) IMPORTANT: After writing the code, you MUST call the execute_code tool with your Python code as the argument.


RULES:
NEVER write tool calls in text. Do NOT output <function=...> ... </function>.
When you need to use a tool, use the available tools [execute_code].
If you output <function=...>, that is an error.

"""


def fix_generated_code(state: WeatherAnalysis) -> dict:
    """Adjust code based on error message and regenerate"""
    MAX_RETRIES = 3
    
    # Check if we've exceeded max retries
    if state.execution_retries >= MAX_RETRIES:
        return {
            "Error": "Unable to generate working code after 3 attempts",
            "analysis": None,
            "anomaly": None,
            "stats": None
        }
    
    # Create a message asking to fix the code based on the error
    error_msg = state.last_error or "Unknown error"
    adjustment_prompt = f"""
    The previous code execution failed with this error: {error_msg}

    1. Fix the issue with this{state.generated_code}
    2. you have access to all pythons Library.
    3. Return the corrected code

    Generate the corrected Python code and call execute_code tool with it.
    """
    
    # Add the error message to conversation
    msgs = state.messages + [HumanMessage(content=adjustment_prompt)]
    
    # Bind tools and generate adjusted code
    llm_with_tools = llm.bind_tools([execute_code])
    result = llm_with_tools.invoke(msgs)
    
    return {
        "messages": msgs + [result],
        "execution_retries": state.execution_retries + 1,
        "generated_code":json.dumps(result.tool_calls[0]['args'].get('code', ''))
    }

def process_execution_results(state: WeatherAnalysis) -> dict:
    """Process the results from execute_code tool execution"""
    try:
        # Get the last tool message (result from execute_code)
        if not state.messages:
            return {"Error": "No messages found"}
        
        # Find the last ToolMessage
        tool_messages = [msg for msg in state.messages if isinstance(msg, ToolMessage)]
        if not tool_messages:
            return {"Error": "No tool execution results found"}
        
        last_tool_msg = tool_messages[-1]
        
        # Parse the JSON result from the tool
        result_dict = json.loads(last_tool_msg.content)
        
        # Check for errors
        if result_dict.get("ok") is False or "error" in result_dict:
            error_msg = result_dict.get("error", "Unknown error in code execution")
            
            # Check if we should retry
            if state.execution_retries < state.max_execution_retries:
                return {
                    "last_error": error_msg,
                    "execution_retries": state.execution_retries + 1
                    # Don't set Error - this will route to adjust_code
                }
            else:
                # Max retries exceeded
                return {
                    "Error": f"Unable to generate working code after {state.max_execution_retries} attempts. Last error: {error_msg}",
                    "analysis": None,
                    "anomaly": None,
                    "stats": None
                }
        
        # Success - validate and extract results
        outcome = WeatherAnalysisOutPut.model_validate(result_dict)
        return {
            "analysis": outcome.analysis,
            "anomaly": outcome.anomaly,
            "stats": outcome.stats,
            "Error": None,
            "execution_retries": 0  # Reset on success
        }
    except Exception as e:
        error_msg = f"Failed to process execution results: {e}"
        if state.execution_retries < state.max_execution_retries:
            return {
                "last_error": error_msg,
                "execution_retries": state.execution_retries + 1
            }
        else:
            return {
                "Error": f"Unable to generate code after {state.max_execution_retries} attempts. {error_msg}",
                "analysis": None,
                "anomaly": None,
                "stats": None
            }

def generate_code_node(state: WeatherAnalysis) -> dict:
    try:
        """Generate Python code based on the historical weather baseline and return the generated code."""
        # Get messages from state, or create initial messages if empty
        if not state.messages:
            msgs= [
            SystemMessage(content=CODEACT_PROMPT),
            HumanMessage(content=f" Understand the data stored in :{state.hist_file_path} and generate Python code, no comments .")
        ]

        else:
            msgs = state.messages

        # Bind the tool to the LLM so it can make tool calls
        llm_with_tools = llm.bind_tools([execute_code])

        result = llm_with_tools.invoke(msgs)
        updated_messages = msgs + [result]
        return {
            "messages": updated_messages,
            "generated_code":json.dumps(result.tool_calls[0]['args'].get('code', '')),
        }
    except Exception as e:
        error_msg = f"Failed at generate code: {e}"
        return {
                "last_error": error_msg,
        }

    
    
import contextlib
import io
@tool
def execute_code(code: str) -> str:
    """
    Execute Python code and return the results as a JSON string.
    
    The code should set a RESULT variable as a dictionary with analysis results.
    Returns the RESULT as a JSON string, or an error message if execution fails.
    You have access to the math, pandas.
    
    Args:
        code: The Python code to execute (as a string)
    
    Returns:
        JSON string containing the RESULT dictionary or error information
    """
    try:
        # Restrict globals for safe execution
        env = {
            "__builtins__": {
                "__import__": builtins.__import__,
                "print": print,
                "range": range,
            },
            "pd": pd,
            "np": np,
            "datetime": datetime,
        }

        # Execute the code
        exec(code, env, env)
        
        # Get result from RESULT variable
        if "RESULT" not in env:
            return json.dumps({"error": "RESULT variable not found in code"})

        return json.dumps(env["RESULT"])
    except Exception as e:
        return json.dumps({"error": str(e)})

# Add this new prompt if 
# Add this function before subgraph2 definition (around line 673)
def compute_baseline_directly(state: WeatherAnalysis) -> dict:
    """After 3 retries failed, ask LLM to compute baseline directly from the file path"""
    try:
        analysis_prompt = f"""Analyze the weather data from this CSV file and compute baseline statistics:
        File path: {state.hist_file_path}
        Read the file, compute baseline statistics, identify anomalies, and provide your analysis."""
        
        # Use LLM with structured output
        structured_llm = llm.with_structured_output(WeatherAnalysisOutPut)
        
        result = structured_llm.invoke([
            HumanMessage(content=analysis_prompt)
        ])
        
        return {
            "analysis": result.analysis,
            "anomaly": result.anomaly,
            "stats": result.stats,
            "Error": None
        }
        
    except Exception as e:
        error_msg = f"Failed to compute baseline directly: {e}"
        return {
            "Error": error_msg,
            "analysis": None,
            "anomaly": None,
            "stats": None
        }

subgraph2 = StateGraph(WeatherAnalysis)
subgraph2.add_node("create_code", generate_code_node)
subgraph2.add_node("execute_code", ToolNode(tools=[execute_code]))
subgraph2.add_node("process_results", process_execution_results)
subgraph2.add_node("adjust_code", fix_generated_code)
subgraph2.add_node("compute_baseline", compute_baseline_directly)

subgraph2.add_edge(START, "create_code")
subgraph2.add_conditional_edges(
    "create_code",
    lambda state: "execute_code" if (
        state.messages and 
        len(state.messages) > 0 and 
        hasattr(state.messages[-1], 'tool_calls') and 
        state.messages[-1].tool_calls
    ) else END,
    path_map={"execute_code": "execute_code", END: END}
)
subgraph2.add_edge("execute_code", "process_results")
subgraph2.add_conditional_edges(
    "process_results",
    route_after_execution,
    path_map={
        "adjust_code": "adjust_code",  # Retry with adjusted code
        "compute_baseline": "compute_baseline",  # llm do the computation itself
        END: END  # Success
    }
)
subgraph2.add_conditional_edges(
    "adjust_code",
    lambda state: "execute_code" if (
        state.messages and 
        len(state.messages) > 0 and 
        hasattr(state.messages[-1], 'tool_calls') and 
        state.messages[-1].tool_calls
    ) else "error_handler",
    path_map={"execute_code": "execute_code", "compute_baseline": "compute_baseline"}
)
subgraph2.add_edge("compute_baseline", END)

analysis_subgraph = subgraph2.compile()


# Or save as PNG (requires pygraphviz)
try:
    analysis_subgraph.get_graph().draw_mermaid_png(output_file_path="analysis_graph.png")
    print("\nGraph saved as analysis_graph.png")
except Exception as e:
    print(f"\nCould not save PNG (pygraphviz may not be installed): {e}")


    

# -----------------------------
# SUBGRAPH 3
# -----------------------------
#CodeACt to compute and do the analysis


# -----------------------------
# SUBGRAPH 4 Recommendation and Reflection
# -----------------------------
def recommend_for_weather(state: RecommendationState)-> dict:
        try:
            """Recommend activities the graph after getting tools"""
            weather_data = parse_mcp(state.today_weather)
            # Handle case where today_weather might be a string or None
            if weather_data is None:
                return {
                    "Error": "No weather data available",
                    "recommendations": []
                }
            prompt = f"""
            You generate practical weather-based recommendations You have the following details.
            Weather today:
            - Temp (C): {weather_data["temperature"]}
            - is_day: {weather_data["is_day"]}
            - Windspeed (kph): {weather_data["windspeed"]}
            - Windspeed Direction: {weather_data["winddirection"]}
            - interval: {weather_data["interval"]}

            Return concise bullets for:
            1) recommendations
            3) safety_notes
            4) recommendation_summary
            Only use info implied by the weather above.
            """

            out: Recommendation = llm.with_structured_output(Recommendation).invoke(prompt)
            # Update state using attribute access (Pydantic model)
            return {
                "original_recommendations": out
            }
        except Exception as e:
            return {
                "Error": f"Failed to recommend: {e}",
                "original_recommendations": None
            }

#reflect on the recommendations 
def reflect_on_recommendations(state: RecommendationState) -> dict:
    try:
        if state.original_recommendations is None:
            return {
                "Error": "No original recommendations to reflect on"
            }
        orig = state.original_recommendations

        reflection_prompt = f"""
            You are writing reflection notes about the quality of recommendations.
            The weather for today is {state.formated_today_weather}

            Original Recommendations: {orig.recommendations}
            Original Safety Notes: {orig.safety_notes}
            Original Summary: {orig.recommendation_summary}

            Analyze and provide:
            1. reflection_notes: A detailed analysis - what assumptions were made, what could be wrong, what info might be missing (e.g. humidity, precipitation chance), and how confident we are.
            2. improvements: Specific suggestions to improve the recommender (inputs to add, rules to include, personalization ideas).
            3. Analyze the safety notes - are they really safe tips? Is there any reason the results are fake or hallucination?
            4. Generate improved recommendations: recommendations (list of strings), safety_notes (list of strings), recommendation_summary (list of strings)

            Return all fields in the required format. Keep it concise and actionable.

            """
        print(reflection_prompt)

        out: Reflections = llm.with_structured_output(Reflections).invoke(reflection_prompt)
              # Create a message asking user if they want to apply the reflection
        reflection_message = f"""
                            Reflection completed. Here's what was improved:

                            Reflection Notes: {out.reflection_notes}
                            Improvements: {out.improvements}

                            Original Recommendations: {orig.recommendation_summary}
                            Improved Recommendations: {out.recommendation_summary}

                            Would you like to apply the improved recommendations? (yes/no)
                                    """
            
        return {
                "reflected_recommendations": out,
                "reflection_notes": out.reflection_notes,
                "messages": state.messages + [HumanMessage(content=reflection_message)]
        }

    except Exception as e:
        error_msg = f"Failed at reflection node: {e}"
        return {
            "Error": error_msg,
            "recommendation_summary": []
        }
    
def ask_apply(state: RecommendationState) -> dict:
    try:
        if state.reflected_recommendations is None or state.original_recommendations is None:
            # If no reflection, just use original
            return {
                "final_recommendations": state.original_recommendations.recommendation_summary if state.original_recommendations else []
            }
        
        # Display the reflection message/question if available
        if state.messages:
            last_message = state.messages[-1]
            if hasattr(last_message, 'content'):
                print("\n" + "="*60)
                print(last_message.content)
                print("="*60)
        
        # Ask user for input
        while True:
            user_input = input("\nDo you want to apply the improved recommendations? (Yes/No): ").strip()
            
            if user_input.lower() in ['yes', 'y']:
                user_decision = "apply"
                final_recs = state.reflected_recommendations.recommendation_summary
                decision_text = "Applied improved recommendations"
                break
            elif user_input.lower() in ['no', 'n']:
                user_decision = "keep_original"
                final_recs = state.original_recommendations.recommendation_summary
                decision_text = "Kept original recommendations"
                break
            else:
                print("Please enter 'Yes' or 'No'")
        
        # Ensure messages list exists
        current_messages = state.messages if state.messages else []
        
        return {
            "final_recommendations": final_recs,
            "user_decision": user_decision,
            "messages": current_messages + [HumanMessage(content=decision_text)]
        }
    except Exception as e:
        return {
            "Error": f"Failed to apply decision: {e}",
            "final_recommendations": state.original_recommendations.recommendation_summary if state.original_recommendations else []
        }


subgraph3 = StateGraph(RecommendationState)
subgraph3.add_node("recommend", recommend_for_weather)
subgraph3.add_node("reflect", reflect_on_recommendations)
subgraph3.add_node("ask_user", ask_apply)

subgraph3.add_edge(START, "recommend")
subgraph3.add_edge("recommend", "reflect")
subgraph3.add_edge("reflect", "ask_user")
subgraph3.add_edge("ask_user", END)
recommend_graph = subgraph3.compile()
# Or save as PNG (requires pygraphviz)
try:
    recommend_graph.get_graph().draw_mermaid_png(output_file_path="recommend_graph.png")
    print("\nGraph saved as recommend_graph.png")
except Exception as e:
    print(f"\nCould not save PNG (pygraphviz may not be installed): {e}")





# -----------------------------
# Starting Point of the code
# -----------------------------
async def main():
    compiled_graph = await create_graph()
    # Example usage - you'll want to modify this based on your needs
    # Or save as PNG (requires pygraphviz)
    try:
        compiled_graph.get_graph().draw_mermaid_png(output_file_path="weather_graph.png")
        print("\nGraph saved as weather_graph.png")
    except Exception as e:
        print(f"\nCould not save PNG (pygraphviz may not be installed): {e}")
    messages = [SystemMessage(content=system_prompt),
            HumanMessage(content="What's the weather today?")]
    result = await compiled_graph.ainvoke({"messages": messages})
# -----------------------------
# Starting Point of the code
# -----------------------------

if __name__ == "__main__":
    asyncio.run(main())