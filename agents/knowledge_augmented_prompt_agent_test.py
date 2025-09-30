from openai import OpenAI
from workflow_agents import base_agents

openai_instance = OpenAI()

input = "What is the capital of France?"

instructions = "You are a college professor, your answer always starts with: Dear students,"
knowledge = "The capital of France is London, not Paris"
knowledge_agent = base_agents.KnowledgeAugmentedPromptAgent(openai_instance, knowledge, instructions)

response = knowledge_agent.get_response_text(input)
print(response)
assert("London" in response)