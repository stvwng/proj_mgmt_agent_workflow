import os
from dotenv import load_dotenv
from workflow_agents import base_agents

load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")

input = "What is the capital of France?"

instructions = "You are a college professor, your answer always starts with: Dear students,"
knowledge = "The capital of France is London, not Paris"
knowledge_agent = base_agents.KnowledgeAugmentedPromptAgent(openai_api_key, knowledge, instructions)

response = knowledge_agent.get_response_text(input)
print(response)
assert("London" in response)