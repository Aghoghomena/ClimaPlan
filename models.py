from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, List, Annotated
from langchain.messages import AnyMessage
from langgraph.graph.message import add_messages





class UserLocationSchema(BaseModel):
    latitude: float = Field(..., ge=-90, le=90)
    longitude: float = Field(..., ge=-180, le=180)
    city: str | None = None
    country: str | None = None


class WeatherAnalysisOutPut(BaseModel):
    analysis: str
    anomaly: Dict[str, Any]  # e.g. {"temp_anomaly_c": 1.2, "precip_anomaly_mm": -0.3, ...}
    stats: Dict[str, Any]


class WeatherAnalysis(BaseModel):
    analysis: str | None = None
    anomaly: Dict[str, Any] = None  # e.g. {"temp_anomaly_c": 1.2, "precip_anomaly_mm": -0.3, ...}
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


#Contains all the classes used
class WeatherState(BaseModel):
    retries: int = 0
    messages: Annotated[list[AnyMessage], add_messages] = []
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
    anomaly: Optional[Dict[str, Any]] = None
    stats: Optional[Dict[str, Any]] = None
    recommendations: Optional[List[str]] = None
    final_output: str | None = None
    # final strict output (as dict)
    final: Optional[Dict[str, Any]] = None
    recommendation_state: Optional[RecommendationState] = None  # Add RecommendationState

class CorrectnessEvalSchema(BaseModel):
    """Schema for code quality evaluation."""
    score: int = Field(description="An integer percentage score from 0 to 100 indicating correctness/quality.")
    comment: str = Field(description="An extensive explanation of the score.")
    failingscore: int = Field(description="0 to 100 indicating likelihood the code will fail.")


class ReflectionQualitySchema(BaseModel):
    score: int = Field(description="0-100 score for whether reflection improves the recommendation.")
    comment: str = Field(description="Why this score; what improved/worsened; any issues.")