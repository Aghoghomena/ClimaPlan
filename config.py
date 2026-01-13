import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_mcp_adapters.client import MultiServerMCPClient

# Load environment variables
load_dotenv()

# Get environment variables
api_key = os.getenv("GEMINI_API_KEY")
api_key2 = os.getenv("GEMINI_API_KEY2")
api_base = os.getenv("GEMINI_API_BASE")
model = os.getenv("GEMINI_API_MODEL")
groq_api_key = os.getenv("GROQ_API_KEY")

# Validate required environment variables (optional - can be removed if validation not needed)
if not all([api_key, model, api_key2]):
    # Note: This validation might not be needed if using Groq
    pass

# Initialize the LLM
# Currently using Groq, but you can switch by uncommenting the ChatGoogleGenerativeAI version

# Option 1: Groq LLM (currently active)
llm = ChatGroq(
    model="llama-3.1-8b-instant",
    groq_api_key=groq_api_key,
    temperature=0.2
)

# Option 2: Google Generative AI LLM (commented out)
# llm = ChatGoogleGenerativeAI(
#     model=model,
#     google_api_key=api_key,
#     temperature=0
# )

mcp_client = MultiServerMCPClient(
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