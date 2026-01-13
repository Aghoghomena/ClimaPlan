# from ast import main
# from xmlrpc import client
# import sys as system
# from langchain_mcp_adapters.client import MultiServerMCPClient
# import asyncio

# mcp_client  = MultiServerMCPClient(
#     {
#         "climate_report_srv": {
#             "transport": "stdio",
#             "command": "uv",
#             "args": ["run", "python", "tools.py"],
#         },
#          "open_meteo": {
#             "transport": "stdio",
#             "command": "npx",
#             "args": ["open-meteo-mcp-server"],
#             "env": {
#                 "OPEN_METEO_ARCHIVE_API_URL": "https://archive-api.open-meteo.com/v1/archive",
#             }
#         }

#     }
# )
# async def get_tools():
#     alltools = await mcp_client.get_tools()
#     print(f"Found {len(alltools)} total tools:")
#     ALLOW = {"weather_archive", "geocoding", "get_user_location", "get_weather_for_today"}  # only what you need
#     tools= [t for t in alltools if t.name in ALLOW]
#     print(f"filtered {len(tools)} total tools:")

#     print(tools)

# if __name__ == "__main__":
#     asyncio.run(get_tools())





from prompts import format_weather_prompt

print(format_weather_prompt("test"), )