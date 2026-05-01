import yaml
import mlflow
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain.agents import create_agent
from langchain_openai import ChatOpenAI
import os

def get_prompt(prompt_name):
    try:
        mlflow.genai.load_prompt(f"prompts:/{prompt_name}@latest")
    except:
        with open(f"prompts/{prompt_name}.yaml") as f:
            new_prompt = list(yaml.safe_load(f))
            mlflow.genai.register_prompt(prompt_name, new_prompt)

    return mlflow.genai.load_prompt(f"prompts:/{prompt_name}@latest")

@mlflow.trace
def process_question(question):
    model = ChatOpenAI(base_url=os.environ["OPENAI_API_BASE"],
                       api_key=os.environ["OPENAI_API_KEY"],
                       model="qwen/qwen3-4b-2507")
    
    prompt = get_prompt("initial_prompt")
    
    agent = create_agent(model=model)
    result = agent.invoke({
        "messages": prompt.format(question=question)
    })

    return result['messages'][-1].content