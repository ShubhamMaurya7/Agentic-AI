import os
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain_core.tools import Tool

from langchain_community.tools import DuckDuckGoSearchRun
from langchain_community.utilities import WikipediaAPIWrapper

from langchain.agents import AgentExecutor, create_react_agent
from langchain import hub

# Load env
load_dotenv()

# LLM (OpenRouter)
llm = ChatOpenAI(
    model="openai/gpt-4o-mini",
    temperature=0,
    base_url=os.getenv("OPENAI_API_BASE"),
    api_key=os.getenv("OPENAI_API_KEY")
)

# Tools
search = DuckDuckGoSearchRun()
wiki = WikipediaAPIWrapper()

tools = [
    Tool(
        name="Search",
        func=search.run,
        description="Search the web"
    ),
    Tool(
        name="Wikipedia",
        func=wiki.run,
        description="Get information from Wikipedia"
    )
]

# Prompt
prompt = hub.pull("hwchase17/react")

# Agent
agent = create_react_agent(llm, tools, prompt)

agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True
)

# Function
def generate_report(topic):
    query = f"""
    Perform detailed research on: {topic}

    Generate a structured report with:
    Cover Page
    Title
    Introduction
    Key Findings
    Challenges
    Future Scope
    Conclusion
    """
    response = agent_executor.invoke({"input": query})
    return response["output"]

# Run
if __name__ == "__main__":
    topic = input("Enter topic: ")
    print("\n===== FINAL REPORT =====\n")
    print(generate_report(topic))