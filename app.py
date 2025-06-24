import dotenv
dotenv.load_dotenv()

import os
import asyncio
from rich import print

from langchain_openai import ChatOpenAI
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver
from prompt_toolkit import PromptSession

OPENAI_KEY = os.environ.get("OPENAI_KEY")

MCP_SERVERS = {}

async def main():
  try:
    model = ChatOpenAI(model="gpt-4-turbo", api_key=OPENAI_KEY)
    client = MultiServerMCPClient(MCP_SERVERS)
    tools = await client.get_tools()
    print('tools', tools)

    memory = MemorySaver()
    agent = create_react_agent(model, tools, checkpointer=memory)
    config = {"configurable": {"thread_id": "1"}}
    session = PromptSession()

    while True:
      user_input = await session.prompt_async("You: ")
      if user_input.lower() in ['exit', 'quit']:
        break
      response = await agent.ainvoke({"messages": user_input}, config=config)
      if response['messages']:
        print(f"Agent: {response['messages'][-1].content}\n\n")
  except KeyboardInterrupt:
    print("\nExiting...")
    return
  except Exception as e:
    import traceback
    traceback.print_exc()
    print(f"Error: {e}")

if __name__ == "__main__":
  try:
    asyncio.run(main())
  except (KeyboardInterrupt, EOFError):
    print("\nExiting...")
  except Exception as e:
    print(f"Application error: {e}")
