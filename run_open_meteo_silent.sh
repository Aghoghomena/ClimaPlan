#!/usr/bin/env bash
set -e

# Run Open-Meteo MCP server
# Keep stdout for MCP protocol
# Suppress noisy logs on stderr

exec npx open-meteo-mcp-server 2>/dev/null
