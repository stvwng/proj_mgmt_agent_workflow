
from openai import OpenAI
from workflow_agents.base_agents import KnowledgeAugmentedPromptAgent, RoutingAgent

openai_instance = OpenAI()

general_knowledge_agent_instructions = "You are a college professor"

texas_knowledge = "You know everything about Texas"
texas_agent = KnowledgeAugmentedPromptAgent(openai_instance, texas_knowledge, general_knowledge_agent_instructions)

europe_knowledge = "You know everything about Europe"
europe_agent = KnowledgeAugmentedPromptAgent(openai_instance, europe_knowledge, general_knowledge_agent_instructions)

math_agent_instructions = "You are a college math professor"
math_knowledge = "You know everything about math, you take prompts with numbers, extract math formulas, and show the answer without explanation"
math_agent = KnowledgeAugmentedPromptAgent(openai_instance, math_knowledge, math_agent_instructions)

agents = [
    {
        "name": "texas agent",
        "description": "Answer a question about Texas",
        "func": lambda x: texas_agent.get_response_text(x)
    },
    {
        "name": "europe agent",
        "description": "Answer a question about Europe",
        "func": lambda x: europe_agent.get_response_text(x)
    },
    {
        "name": "math agent",
        "description": "When a prompt contains numbers, respond with a math formula",
        "func": lambda x: math_agent.get_response_text(x)
    }
]

routing_agent = RoutingAgent(openai_instance, agents)

inputs = [
    "Tell me about the history of Rome, Texas",
    "Tell me about the history of Rome, Italy",
    "One story takes 2 days, and there are 20 stories"
]

for i in inputs:
    print(routing_agent.route_prompt(i))