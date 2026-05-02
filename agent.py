import yaml
import mlflow
from langchain.agents import create_agent
from langgraph.graph import MessagesState
from langchain_openai import ChatOpenAI
from langchain_community.tools import DuckDuckGoSearchResults
from langchain_community.tools.playwright.utils import create_async_playwright_browser
from langchain_community.agent_toolkits import PlayWrightBrowserToolkit
import os
import json
import time
from contextlib import AsyncContextDecorator

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

class Agent(AsyncContextDecorator):
    async def __aenter__(self):
        model = ChatOpenAI(base_url=os.environ["OPENAI_BASE_URL"],
                       api_key=os.environ["OPENAI_API_KEY"],
                       model=os.environ["OPENAI_MODEL"])
        self.__browser = create_async_playwright_browser(headless=False)
        #tools = [DuckDuckGoSearchResults() ]
        self.__prompt = get_prompt("initial_prompt")
        self.__agent = create_agent(model=model, tools=None)
        return self
    
    async def __aexit__(self, exc_type, exc_value, traceback):
        for context in self.__browser.contexts:
            await context.close()
        return False

    async def ainvoke(self, question: str):
        result = await self.__agent.ainvoke({
            "messages": self.__prompt.format(question=question)
        },config={"recursion_limit": 5})

        return result['messages'][-1].content.strip()

@mlflow.trace
async def process_question(question):
    async with Agent() as agent:
        return await agent.ainvoke(question)
