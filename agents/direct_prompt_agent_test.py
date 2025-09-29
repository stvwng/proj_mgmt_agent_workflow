# Test script for DirectPromptAgent class

from workflow_agents import base_agents
import os
from dotenv import load_dotenv


load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

input = "What is the Capital of France?"

direct_prompt_agent = base_agents.PromptAgent(OPENAI_API_KEY)
direct_agent_response = direct_prompt_agent.get_response_text(input)

print(direct_agent_response)
print("Agent generated the response using general knowledge from the LLM.")
