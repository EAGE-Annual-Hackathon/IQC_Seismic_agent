#!/usr/bin/env python3
"""
IQC Assistant LLM Agent
"""

import os
import asyncio

from pydantic_ai import Agent
from pydantic_ai.mcp import MCPServerHTTP
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.providers.openai import OpenAIProvider

from config import BASE_PATH, PREPROMPT_PATH, API_KEY, MCP_SERVER_URL, INTERFACE_SERVER_URL

os.environ['OPENAI_API_KEY'] = API_KEY

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise RuntimeError("Please set the OPENAI_API_KEY environment variable.")

try:
    with open(PREPROMPT_PATH, "r", encoding="utf-8") as f:
        IQC_PREPROMPT = f.read()
except FileNotFoundError:
    print("⚠️  Warning: preprompt.md not found. Using no preprompt.")
    IQC_PREPROMPT = ''

mcp = MCPServerHTTP(url=MCP_SERVER_URL)
agent = Agent(
    model="openai:gpt-4.1-mini",
    mcp_servers=[mcp],
    system_prompt=IQC_PREPROMPT
)

async def _ask_agent_async(question: str) -> str:
    async with agent.run_mcp_servers():
        response = await agent.run(question)
        return response.output

def ask_agent_sync(question: str) -> str:
    return asyncio.run(_ask_agent_async(question))
