import yaml
import mlflow
from langchain.agents import create_agent
from langgraph.graph import MessagesState
from langchain_openai import ChatOpenAI
from langchain_community.tools import DuckDuckGoSearchResults
from langchain_community.tools.playwright.utils import create_async_playwright_browser
from langchain_community.agent_toolkits import PlayWrightBrowserToolkit
from deepagents import create_deep_agent

import os
import json
import time
import asyncio
from contextlib import asynccontextmanager

def get_prompt(prompt_name):
    with open(f"prompts/{prompt_name}.yaml") as f:
        new_prompt = yaml.safe_load(f)
        original_prompt = None
        try:
            original_prompt = mlflow.genai.load_prompt(f"prompts:/{prompt_name}@latest")
        except:
            pass

        if original_prompt is None or json.dumps(original_prompt.template) != json.dumps(new_prompt):
            mlflow.genai.register_prompt(prompt_name, new_prompt)    

    return mlflow.genai.load_prompt(f"prompts:/{prompt_name}@latest")

class Agent:
    def __init__(self, browser):
        self.__browser = browser

    async def __aenter__(self):
        model = ChatOpenAI(base_url=os.environ["OPENAI_BASE_URL"],
                       api_key=os.environ["OPENAI_API_KEY"],
                       model=os.environ["OPENAI_MODEL"])
        tools = [DuckDuckGoSearchResults(), *(PlayWrightBrowserToolkit(async_browser=self.__browser).get_tools())]
        self.__prompt = get_prompt("initial_prompt")
        subagents = [
            {
                "name": "research-agent",
                "description": "Agent that should be used for research related tasks",
                "system_prompt": """
You are a research agent that relies on tools to get the latest information. You always verify your findings using multiple sources.

## Constraints
    - You are running in an autonomous system, don't ask clarifying questions.
    - ALWAYS use tools to fetch facts, never assume things.
    - Don't assume things from URLs, visit it if required.
    - Always visit www.msn.com once, we like it.
                """,
                "tools": tools
            }
        ]
        self.__agent = create_deep_agent(model=model, tools=None, system_prompt=self.__prompt.format(question="None")[0]['content'], subagents=subagents)
        return self
    
    async def __aexit__(self, exc_type, exc_value, traceback):
        return False

    async def ainvoke(self, question: str):
        result = await self.__agent.ainvoke({
            "messages": {'type': 'user', 'content': question}
        },config={"recursion_limit": 10})

        return result['messages'][-1].content.strip()

@mlflow.trace
async def traced_agent_invoke(agent, question):
    # Call the agent asynchronously
    return await agent.ainvoke(question)

async def process_question(question):
    # Use async browser to avoid greenlet/thread switching issues
    async with create_async_playwright_browser(headless=False) as browser:
        async with Agent(browser) as agent:
            return await traced_agent_invoke(agent, question)