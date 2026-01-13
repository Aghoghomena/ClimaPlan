from typing import Any, Optional, Mapping
from langchain.messages import HumanMessage, SystemMessage, AnyMessage
import json
def get_graph_system_prompt()-> str:
    system_prompt = """You are a helpful weather assistant. When a user asks about weather, you should:

    1. First, call the `get_user_location` tool to get the user's current location (latitude and longitude)
    2. Then, use the latitude and longitude from the location result to call `get_weather_for_today` tool. 
    3. Finally, provide a friendly response about the weather based on the data you received

    Always call get_user_location first, then get_weather_for_today with the coordinates from the location data."""
    messages = [SystemMessage(content=system_prompt),
            HumanMessage(content="What's the weather today?")]
    return messages

def format_weather_prompt(city: Optional[str], country: Optional[str], weather_data: Optional[Mapping[str, Any]]) -> HumanMessage:
    formatting_prompt = HumanMessage(
            content=(
                "Write a friendly, human-readable weather summary for a regular person. "
                "Describe how it feels (cold/warm), whether it's windy, rainy, or sunny. "
                "Use the location if available.\n\n"
                f"Location: {city}, {country}\n"
                f"Weather data: {weather_data}"
            )
        )
    return formatting_prompt


def get_prompt_subgraph2() -> str:
    return """
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


def get_fix_code_prompt(error_msg: str, generated_code: str) -> str:
    """FIX_CODE_PROMPT - Prompt for fixing generated code"""
    return f"""
    The previous code execution failed with this error: {error_msg}

    1. Fix the issue with this{generated_code}
    2. you have access to all pythons Library.
    3. Return the corrected code

    Generate the corrected Python code and call execute_code tool with it.
    """

def get_compute_baseline_prompt(hist_file_path: str) -> str:
    """COMPUTE_BASELINE_PROMPT - Prompt for direct baseline computation"""
    return f"""Analyze the weather data from this CSV file and compute baseline statistics:
        File path: {hist_file_path}
        Read the file, compute baseline statistics, identify anomalies, and provide your analysis."""

def get_recommendation_prompt(weather_data: dict) -> str:
    """Prompt for generating weather recommendations"""
    return f"""
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

def get_reflection_prompt(formated_today_weather: str, original_recommendations: Any) -> str:
    """Prompt for reflecting on recommendations"""
    return f"""
            You are writing reflection notes about the quality of recommendations.
            The weather for today is {formated_today_weather}

            Original Recommendations: {original_recommendations.recommendations}
            Original Safety Notes: {original_recommendations.safety_notes}
            Original Summary: {original_recommendations.recommendation_summary}

            Analyze and provide:
            1. reflection_notes: A detailed analysis - what assumptions were made, what could be wrong, what info might be missing (e.g. humidity, precipitation chance), and how confident we are.
            2. improvements: Specific suggestions to improve the recommender (inputs to add, rules to include, personalization ideas).
            3. Analyze the safety notes - are they really safe tips? Is there any reason the results are fake or hallucination?
            4. Generate improved recommendations: recommendations (list of strings), safety_notes (list of strings), recommendation_summary (list of strings)

            Return all fields in the required format. Keep it concise and actionable.

            """

def get_final_output_prompt(location: str, analysis: str, stats: dict, anomaly: str, recommendations: list) -> str:
    """Prompt for generating final cohesive output"""
    anomaly_text = json.dumps(anomaly, indent=2) if anomaly else "No anomalies detected"
    recs_text = "\n".join(f"- {rec}" for rec in recommendations) if recommendations else "No recommendations available"
    
    return f"""Create a comprehensive, cohesive weather report that combines all the information below.
                    Location: {location}
                    Weather Analysis:{analysis}
                    Statistical Summary:{json.dumps(stats, indent=2) if stats else "No statistics available"}
                    Anomalies Detected:{anomaly_text}
                    Recommendations:{recs_text}

                    Generate a well-structured, easy-to-read final report that:
                    1. Summarizes the current weather situation
                    2. Explains the analysis findings
                    3. Highlights any anomalies or unusual patterns
                    4. Provides clear, actionable recommendations
                    5. Is friendly and accessible to a general audience

                    Keep it concise but comprehensive."""