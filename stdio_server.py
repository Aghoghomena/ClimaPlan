import asyncio
from langchain_mcp_adapters.client import MultiServerMCPClient

async def main():
    mcp_client = MultiServerMCPClient({
        "open_meteo": {
            "transport": "stdio",
            "command": "npx",
            "args": ["open-meteo-mcp-server"],
            # optional env overrides (these are documented in the repo)
            "env": {
                "OPEN_METEO_API_URL": "https://api.open-meteo.com",
                "OPEN_METEO_ARCHIVE_API_URL": "https://archive-api.open-meteo.com",
                "OPEN_METEO_GEOCODING_API_URL": "https://geocoding-api.open-meteo.com",
            }
        }
    })

    tools = await mcp_client.get_tools()
    print([t.name for t in tools])

asyncio.run(main())
