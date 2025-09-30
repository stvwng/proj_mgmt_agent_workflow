from openai import OpenAI
from agents import base_agents

openai_instance = OpenAI()

input = "What is the Capital of France?"

direct_prompt_agent = base_agents.BaseAgent(openai_instance)
direct_agent_response = direct_prompt_agent.get_response_text(input)

print(direct_agent_response)
print("Agent generated the response using general knowledge from the LLM.")
