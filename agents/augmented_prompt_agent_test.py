from openai import OpenAI
from agent_classes.base_agent import BaseAgent

openai_instance = OpenAI()

input = "What is the capital of France?"
instructions = "You are a college professor; your answers always start with: 'Dear students,'"

augmented_prompt_agent = BaseAgent(openai_instance, instructions)
augmented_agent_response = augmented_prompt_agent.get_response_text(input)
print(augmented_agent_response)

# The agent used the LLM's knowledge to answer the prompt. The instructions modified the answer's format.
# While the instructions assigned a role to the LLM, I don't believe that it affected the substance of the response here,
# as the answer is common knowledge.
