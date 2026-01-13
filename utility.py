import json
from typing import Any, Dict
from typing import Optional, Dict, Any, List
from langchain.messages import ToolMessage
import pandas as pd
from datetime import datetime
from pathlib import Path

BASELINE_PATH = Path("baseline.py")   # adjust if needed
BASELINE_CODE = BASELINE_PATH.read_text(encoding="utf-8")

def _tool_content_to_dict(content: Any) -> Dict[str, Any]:
    if content is None:
        return {}
    if isinstance(content, dict):
        return content
    if isinstance(content, str):
        content = content.strip()
        if not content:
            return {}
        # Try JSON parse
        try:
            return json.loads(content)
        except json.JSONDecodeError:
            # Sometimes tools return plain text; wrap it
            return {"text": content}
    # Fallback
    return {"value": content}


def _get_trailing_tool_messages(messages: List[Any]) -> List[ToolMessage]:
    """ToolNode may append multiple ToolMessages; we grab the consecutive block at the end."""
    trailing = []
    for msg in reversed(messages):
        if isinstance(msg, ToolMessage):
            trailing.append(msg)
        else:
            break
    return list(reversed(trailing))


def inc_count(counts: dict[str, int], key: str) -> dict[str, int]:
    if not key: return {}
    count = counts.get(key, 0) + 1
    return {key : count }


def weather_data_to_df(raw: dict) -> pd.DataFrame:
    """
    Converts raw weather data dictionary to a pandas DataFrame and filters to get only required and relevant data.

    Args:
        raw (dict): Raw weather data.

    Returns:
        pd.DataFrame: DataFrame containing weather data.
    """
    try:
        weather_list = []
        daily = raw["daily"]

        for date, tmax, tmin, rain, wind, shortwave in zip(
            daily["time"],
            daily["temperature_2m_max"],
            daily["temperature_2m_min"],
            daily["precipitation_sum"],
            daily["wind_speed_10m_max"],
            daily["shortwave_radiation_sum"],
        ):
            weather_list.append({
                "date": date,
                "temp_max": tmax,
                "temp_min": tmin,
                "precipitation": rain,
                "windspeed": wind,
                "shortwave": shortwave
            })
            # Create DataFrame and get day of year
        df = pd.DataFrame(weather_list)
        print
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values("date")
        df["doy"] = df["date"].dt.dayofyear

        # Create baseline data window
        data_window = 5
        # ✅ Use latest date in the returned dataset, not datetime.now()
        target_doy = int(df["doy"].iloc[-1])
        lower = target_doy - data_window
        upper = target_doy + data_window

        if lower < 1:
            baseline = df[(df["doy"] >= 366 + lower) | (df["doy"] <= upper)]
        elif upper > 366:
            baseline = df[(df["doy"] >= lower) | (df["doy"] <= upper - 366)]
        else:
            baseline = df[(df["doy"] >= lower) & (df["doy"] <= upper)]
        print (baseline)
        return baseline
    except Exception as e:
        return pd.DataFrame({"error": [str(e)]})
    

def _parse_payload(payload: dict) -> dict:
    """
    Your climate_report tool returns: {"value":[{"type":"text","text":"{...json...}"}]}
    This parses that into a dict.
    """
    if isinstance(payload, dict) and "value" in payload:
        text = payload["value"][0]["text"]
        return json.loads(text)
    return payload  # already a dict

#safely parse MCP result
def parse_mcp_text_result(result: Any) -> Dict[str, Any]:
    item = result[0] if isinstance(result, list) else result
    if not isinstance(item, dict) or "text" not in item:
        raise ValueError(f"Unexpected MCP payload shape: {type(result)} -> {item}")
    return json.loads(item["text"])

def parse_mcp(result: Any) -> dict:
    item = result[0] if isinstance(result, list) else result

    # common MCP adapter envelope
    if isinstance(item, dict) and "value" in item:
        item = item["value"][0]

    if isinstance(item, dict) and "text" in item and isinstance(item["text"], str):
        return json.loads(item["text"])

    if isinstance(item, dict):
        return item  # already a dict

    raise TypeError(f"Unexpected MCP result type: {type(result)}")

def get_evaluator_prompt():
    # Prompt to evaluate code performace
    """"
    You are evaluating python code to get the :
    You are evaluating the code:
    {generated_code}
    Reviewing against this benchmark code:
    {expected_code}
    review the code for quality, efficiency, and best practices.:
    Provide a score from 0 to 10, where 10 and would perform better than the benchmark means the code is of excellent quality and efficiency and 0 means the code is of very poor quality and efficiency.:
    Score:
    Provide a brief explanation of the score why you gave that score:
    Comment:
    """


def get_baseline_code():
    return BASELINE_CODE