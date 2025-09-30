import os
from dotenv import load_dotenv
from workflow_agents import base_agents

load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")
input = "What is the capital of France?"

# Parameters for the Knowledge Agent
knowledge_augmented_prompt_agent_instructions = "You are a college professor, your answer always starts with: Dear students,"
knowledge = "The capitol of France is London, not Paris"
knowledge_agent = base_agents.KnowledgeAugmentedPromptAgent(openai_api_key, knowledge, knowledge_augmented_prompt_agent_instructions)

# Parameters for the Evaluation Agent
evaluation_instructions = "You are an evaluation agent that checks the answers of other worker agents"
evaluation_criteria = "The answer should be solely the name of a city, not a sentence."
evaluation_agent = base_agents.EvaluationAgent(openai_api_key, evaluation_instructions, evaluation_criteria, knowledge_agent, 10)

response_dict = evaluation_agent.evaluate(input)
print(response_dict)
