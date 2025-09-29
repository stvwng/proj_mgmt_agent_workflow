# Test script for DirectPromptAgent class

from workflow_agents import base_agents
import os
from dotenv import load_dotenv


load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

input = "What is the Capital of France?"

direct_agent = base_agents.DirectPromptAgent(OPENAI_API_KEY)
direct_agent_response = direct_agent.respond(input)

print(direct_agent_response)
print("Agent generated the response using general knowledge from the LLM.")
